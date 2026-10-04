"""
FaceFuel v4 — Tri-Modal Server
================================
POST /analyze           selfie → face + eye
POST /analyze/tongue    tongue photo
POST /analyze/combined  selfie + tongue → all three modalities
GET  /health
GET  /

v4 changes:
  Face  YOLO: 10 classes  mAP=0.560
  Tongue YOLO: 14 classes mAP=0.871
  Eye   YOLO:  6 classes  mAP=0.990
  Deficiencies: 22 dimensions
"""

import os, sys, time, base64, logging, traceback
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import cv2
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

os.environ["GLOG_minloglevel"] = "2"
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent))

HOST   = "0.0.0.0"
PORT   = 8000
MAX_MB = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── v4 Weight paths ──────────────────────────────────────────
FACE_YOLO_PT   = r"runs\detect\runs\detect\runs\face\face_yolo11m_v4\weights\best.pt"
TONGUE_YOLO_PT = r"runs\detect\training_runs\tongue_v4\weights\best.pt"
EYE_YOLO_PT    = r"runs\detect\training_runs\eye_v2\weights\best.pt"

FACE_MLP_PT    = r"facefuel_models\face_severity_mlp_v4.pt"
TONGUE_MLP_PT  = r"facefuel_models\tongue_severity_mlp_v4.pt"
EYE_MLP_PT     = r"facefuel_models\eye_severity_mlp_v4.pt"

# ── v4 class lists ───────────────────────────────────────────
FACE_CLASSES   = [
    "dark_circle","acne","blackhead","wrinkle","redness",
    "dark_spot","rosacea","vitiligo","eczema","butterfly_rash",
]
TONGUE_CLASSES = [
    "tongue_body","white_coating","yellow_coating","thick_coating",
    "no_coating","red_tongue","pale_tongue","purple_tongue",
    "fissured","geographic","smooth_glossy","crenated",
    "oral_ulcer","angular_stomatitis","median_rhomboid",
    "lichen_planus","leukoplakia","hairy_leukoplakia",
]
EYE_CLASSES    = [
    "conjunctival_pallor","scleral_icterus","xanthelasma",
    "pterygium","conjunctivitis","eyelid_drooping",
]

# ── 22-dimension deficiency framework ───────────────────────
ALL_DEFS = [
    "iron_deficiency","b12_deficiency","vitamin_d_deficiency",
    "zinc_deficiency","omega3_deficiency","vitamin_a_deficiency",
    "vitamin_c_deficiency","poor_sleep_quality","hormonal_imbalance",
    "dehydration","high_stress","liver_stress","gut_dysbiosis",
    "hypothyroid","folate_deficiency","cholesterol_imbalance",
    "riboflavin_deficiency","autoimmune_risk","eye_inflammation",
    "copper_deficiency","skin_inflammation","oral_health_risk",
]

FACE_DEFS   = ALL_DEFS[:11]   # first 11 shared
TONGUE_DEFS = ALL_DEFS[:16]   # adds liver_stress … cholesterol_imbalance
EYE_DEFS    = ALL_DEFS        # all 22

FOOD_RECS = {
    "iron_deficiency":        ["spinach","lentils","red meat","tofu","pumpkin seeds"],
    "b12_deficiency":         ["eggs","dairy","salmon","beef liver","fortified cereals"],
    "vitamin_d_deficiency":   ["fatty fish","egg yolks","fortified milk","mushrooms"],
    "zinc_deficiency":        ["oysters","beef","chickpeas","cashews","pumpkin seeds"],
    "omega3_deficiency":      ["salmon","walnuts","flaxseed","chia seeds","mackerel"],
    "vitamin_a_deficiency":   ["sweet potato","carrots","kale","egg yolks","liver"],
    "vitamin_c_deficiency":   ["citrus fruits","bell peppers","broccoli","kiwi"],
    "poor_sleep_quality":     ["magnesium","tart cherry juice","reduce caffeine"],
    "hormonal_imbalance":     ["healthy fats","fiber","reduce sugar","see doctor"],
    "dehydration":            ["8+ glasses water daily","cucumber","watermelon"],
    "high_stress":            ["meditation","B-complex vitamins","magnesium","exercise"],
    "liver_stress":           ["reduce alcohol","leafy greens","beets","milk thistle"],
    "gut_dysbiosis":          ["probiotics","fermented foods","fiber","reduce sugar"],
    "hypothyroid":            ["iodine-rich foods","selenium","zinc","consult doctor"],
    "folate_deficiency":      ["leafy greens","lentils","asparagus","fortified cereals"],
    "cholesterol_imbalance":  ["oats","beans","avocado","olive oil","salmon"],
    "riboflavin_deficiency":  ["dairy","eggs","lean meat","almonds","spinach"],
    "autoimmune_risk":        ["anti-inflammatory diet","omega-3","vitamin D","consult doctor"],
    "eye_inflammation":       ["omega-3","vitamin A","cold compresses","see doctor"],
    "copper_deficiency":      ["shellfish","nuts","seeds","dark chocolate","legumes"],
    "skin_inflammation":      ["omega-3","zinc","vitamin E","reduce dairy","probiotics"],
    "oral_health_risk":       ["vitamin C","zinc","probiotics","reduce sugar","see dentist"],
}
ADVICE = {
    "iron_deficiency":       "Pair with Vitamin C to boost absorption.",
    "b12_deficiency":        "Mainly from animal sources. Vegans must supplement.",
    "vitamin_d_deficiency":  "15-30 min sunlight daily. Consider D3 in winter.",
    "zinc_deficiency":       "Soak legumes to reduce phytates.",
    "omega3_deficiency":     "Aim for 2 servings fatty fish per week.",
    "vitamin_a_deficiency":  "Fat-soluble — pair with healthy fats.",
    "vitamin_c_deficiency":  "Eat some raw fruits/vegetables daily.",
    "poor_sleep_quality":    "Consistent sleep/wake times. Aim 7-9 hrs.",
    "hormonal_imbalance":    "Requires medical evaluation.",
    "dehydration":           "Aim for pale yellow urine throughout the day.",
    "high_stress":           "Both diet and stress management needed.",
    "liver_stress":          "Reduce processed foods and alcohol.",
    "gut_dysbiosis":         "Increase fiber and fermented foods.",
    "hypothyroid":           "Consult doctor for thyroid testing.",
    "folate_deficiency":     "Especially important during pregnancy.",
    "cholesterol_imbalance": "Reduce saturated fats; increase soluble fiber.",
    "riboflavin_deficiency": "Common in those avoiding dairy and meat.",
    "autoimmune_risk":       "Anti-inflammatory lifestyle and medical review.",
    "eye_inflammation":      "Avoid allergens; keep eyes moist; see doctor.",
    "copper_deficiency":     "Rare — usually related to zinc over-supplementation.",
    "skin_inflammation":     "Address gut health and reduce inflammatory foods.",
    "oral_health_risk":      "Regular dental check-ups and oral hygiene.",
}

# ── Feature → deficiency mapping ────────────────────────────
FACE_FEAT_DEF = {
    "dark_circle":    ["iron_deficiency","b12_deficiency","poor_sleep_quality"],
    "acne":           ["hormonal_imbalance","zinc_deficiency","high_stress"],
    "blackhead":      ["zinc_deficiency","vitamin_a_deficiency"],
    "wrinkle":        ["vitamin_c_deficiency","dehydration","omega3_deficiency"],
    "redness":        ["skin_inflammation","vitamin_c_deficiency","high_stress"],
    "dark_spot":      ["vitamin_c_deficiency","vitamin_a_deficiency"],
    "rosacea":        ["gut_dysbiosis","omega3_deficiency","skin_inflammation"],
    "vitiligo":       ["autoimmune_risk","b12_deficiency","copper_deficiency"],
    "eczema":         ["omega3_deficiency","zinc_deficiency","skin_inflammation"],
    "butterfly_rash": ["autoimmune_risk"],
}
TONGUE_FEAT_DEF = {
    "white_coating":    ["gut_dysbiosis","dehydration"],
    "yellow_coating":   ["liver_stress","gut_dysbiosis"],
    "thick_coating":    ["gut_dysbiosis","liver_stress"],
    "red_tongue":       ["b12_deficiency","folate_deficiency","high_stress"],
    "pale_tongue":      ["iron_deficiency","b12_deficiency"],
    "fissured":         ["b12_deficiency","zinc_deficiency","dehydration"],
    "geographic":       ["zinc_deficiency","vitamin_b_deficiency"],
    "smooth_glossy":    ["b12_deficiency","folate_deficiency"],
    "crenated":         ["hypothyroid","dehydration"],
    "oral_ulcer":       ["b12_deficiency","iron_deficiency","oral_health_risk"],
    "lichen_planus":    ["autoimmune_risk","high_stress","oral_health_risk"],
    "leukoplakia":      ["vitamin_a_deficiency","oral_health_risk"],
    "hairy_leukoplakia":["autoimmune_risk","oral_health_risk"],
}
EYE_FEAT_DEF = {
    "conjunctival_pallor": ["iron_deficiency","b12_deficiency"],
    "scleral_icterus":     ["liver_stress"],
    "xanthelasma":         ["cholesterol_imbalance"],
    "pterygium":           ["vitamin_a_deficiency","dehydration"],
    "conjunctivitis":      ["vitamin_a_deficiency","vitamin_c_deficiency","eye_inflammation"],
    "eyelid_drooping":     ["hypothyroid"],
}

_face_pipeline   = None
_tongue_pipeline = None
_eye_pipeline    = None


def get_face_pipeline():
    global _face_pipeline
    if _face_pipeline is None:
        from step10_inference import get_models
        _face_pipeline = {"models": get_models(DEVICE)}
    return _face_pipeline


def get_tongue_pipeline():
    global _tongue_pipeline
    if _tongue_pipeline is None:
        from tongue_inference import get_tongue_models
        _tongue_pipeline = {"models": get_tongue_models(DEVICE)}
    return _tongue_pipeline


def get_eye_pipeline():
    global _eye_pipeline
    if _eye_pipeline is None:
        from eye_inference import get_eye_models
        _eye_pipeline = {"models": get_eye_models(DEVICE)}
    return _eye_pipeline


# ── Posterior fusion ─────────────────────────────────────────
def fuse_posteriors(face_post, tongue_post, eye_post):
    fw, tw, ew = 0.40, 0.35, 0.25
    def get(post, defs, name):
        return post[defs.index(name)] if name in defs and len(post) > defs.index(name) else 0.0
    fused = {}
    for name in ALL_DEFS:
        fp = get(face_post,   FACE_DEFS,   name)
        tp = get(tongue_post, TONGUE_DEFS, name)
        ep = get(eye_post,    EYE_DEFS,    name)
        has = [(p,w) for p,w in [(fp,fw),(tp,tw),(ep,ew)] if p > 0]
        if   len(has) == 3: fused[name] = (fp**fw)*(tp**tw)*(ep**ew)
        elif len(has) == 2:
            p1,w1=has[0]; p2,w2=has[1]; tw_=w1+w2
            fused[name] = (p1**(w1/tw_))*(p2**(w2/tw_))
        elif len(has) == 1:
            p1,w1=has[0]
            fused[name] = p1*(0.85 if w1 < 0.40 else 1.0)
        else:
            fused[name] = 0.0
    total = sum(fused.values()) + 1e-9
    return {k: v/total for k,v in fused.items()}


def build_response(fused, face_feats, tongue_feats, eye_feats, timing):
    sdef = sorted(fused.items(), key=lambda x: -x[1])
    da   = {name: {
        "probability":     round(p,4),
        "probability_pct": f"{p*100:.1f}%",
        "priority_rank":   r,
        "foods":           FOOD_RECS.get(name,[]),
        "advice":          ADVICE.get(name,""),
        "confidence_band": "high" if p>0.20 else "moderate" if p>0.10 else "low",
    } for r,(name,p) in enumerate(sdef,1)}

    top = [{"rank":r,"issue":name,"probability":f"{p*100:.1f}%",
            "priority":"HIGH" if p>0.20 else "MODERATE" if p>0.10 else "LOW",
            "top_foods":FOOD_RECS.get(name,[])[:3],
            "advice":ADVICE.get(name,"")}
           for r,(name,p) in enumerate(sdef[:5],1) if p>0.08]

    return {"deficiency_analysis":da, "top_insights":top,
            "timing_ms":{k:round(v*1000,1) if isinstance(v,float) else v
                         for k,v in timing.items()}}


async def decode_img(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if len(data) > MAX_MB*1024*1024:
        raise HTTPException(413, f"Max {MAX_MB}MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Cannot decode image")
    return img

def enc(bgr, size=(300,300)):
    _,buf = cv2.imencode(".jpg", cv2.resize(bgr,size),
                          [cv2.IMWRITE_JPEG_QUALITY,88])
    return base64.b64encode(buf).decode()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"\n{'='*58}")
    print(f"  FaceFuel v4 — Tri-Modal Server")
    print(f"  Device: {DEVICE}" +
          (f"  GPU: {torch.cuda.get_device_name(0)}" if DEVICE=="cuda" else ""))
    print(f"  22 deficiency dimensions  |  3 modalities")
    print(f"{'='*58}")
    for name, loader in [("Face",  get_face_pipeline),
                          ("Tongue",get_tongue_pipeline),
                          ("Eye",   get_eye_pipeline)]:
        try:   loader(); print(f"  ✅ {name} ready")
        except Exception as e: print(f"  ⚠ {name}: {e}")
    print("  Server ready.\n")
    yield


app = FastAPI(title="FaceFuel v4", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["GET","POST"], allow_headers=["*"])
STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
async def frontend():
    idx = STATIC / "index.html"
    return HTMLResponse(idx.read_text(encoding="utf-8") if idx.exists()
                        else "<h1>FaceFuel v4</h1>")


@app.get("/health")
async def health():
    gpu = {"name": torch.cuda.get_device_name(0)} if DEVICE=="cuda" else {}
    return {"status":"healthy","version":"4.0.0","device":DEVICE,"gpu":gpu,
            "face_classes":  len(FACE_CLASSES),
            "tongue_classes":len(TONGUE_CLASSES),
            "eye_classes":   len(EYE_CLASSES),
            "deficiencies":  len(ALL_DEFS),
            "models":{
                "face_yolo":  f"YOLO11m 10-class mAP=0.560 {FACE_YOLO_PT}",
                "tongue_yolo":f"YOLO11m 14-class mAP=0.871 {TONGUE_YOLO_PT}",
                "eye_yolo":   f"YOLO11m  6-class mAP=0.990 {EYE_YOLO_PT}",
            }}


@app.post("/analyze")
async def analyze_face(file: UploadFile = File(...)):
    img = await decode_img(file)
    try:
        from step10_inference import (preprocess_image, analyze_color_features,
            run_yolo, extract_dinov2_features, run_severity_mlp,
            bayesian_inference_v2, format_output)
        from eye_inference import run_eye_inference

        fm = get_face_pipeline()["models"]; t = {}
        ts=time.time(); prep=preprocess_image(img,fm); t["preprocess"]=time.time()-ts
        if prep is None:
            return JSONResponse({"status":"no_face_detected",
                "message":"No face detected. Use a clear front-facing photo."})

        ts=time.time(); cf=analyze_color_features(prep["raw_rgb"],prep["regions"]); t["color"]=time.time()-ts
        ts=time.time(); yd,yc=run_yolo(prep["aligned_bgr"],fm); t["yolo_face"]=time.time()-ts
        ts=time.time(); fv=extract_dinov2_features(prep["regions"],fm); t["dinov2"]=time.time()-ts
        ts=time.time(); sv,un=run_severity_mlp(fv,fm); t["severity"]=time.time()-ts
        ts=time.time(); fp=bayesian_inference_v2(sv,un,cf,yc); t["bayes"]=time.time()-ts
        face_result = format_output(sv,un,yd,yc,fp,cf,t)

        ts=time.time(); eye_result=run_eye_inference(prep["aligned_bgr"],DEVICE); t["eye"]=time.time()-ts

        fused = fuse_posteriors(face_result.get("posterior",[0]*11),
                                [0.0]*16, eye_result.get("posterior",[0]*22))
        out   = build_response(fused,
                               face_result.get("features_detected",{}),
                               {}, eye_result.get("features_detected",{}), t)

        return JSONResponse({"status":"success","modality":"face+eye",
            "face_features":face_result.get("features_detected",{}),
            "eye_features": eye_result.get("features_detected",{}),
            "tongue_features":{}, **out,
            "aligned_face_b64":enc(prep["aligned_bgr"]),
            "disclaimer":"FaceFuel v4 provides wellness awareness only — not medical diagnosis."})
    except Exception as e:
        print(f"[ERROR] /analyze: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Analysis failed: {e}")


@app.post("/analyze/tongue")
async def analyze_tongue(file: UploadFile = File(...)):
    img = await decode_img(file)
    try:
        from tongue_inference import (detect_and_crop_tongue, run_tongue_yolo,
            extract_tongue_features, run_tongue_severity, tongue_bayesian_inference,
            TONGUE_FEATURES, DEFICIENCIES as T_DEFS, FOOD_RECS as T_FOOD,
            SMALL_LESION_FEATS, count_to_severity)
        m = get_tongue_pipeline()["models"]; t = {}
        ts=time.time(); crop=detect_and_crop_tongue(img,m); t["crop"]=time.time()-ts
        ts=time.time(); yd,yc=run_tongue_yolo(crop,m); t["yolo"]=time.time()-ts
        ts=time.time(); fv=extract_tongue_features(crop,m); t["dinov2"]=time.time()-ts
        ts=time.time(); sv,un=run_tongue_severity(fv,m); t["severity"]=time.time()-ts
        ts=time.time(); tp=tongue_bayesian_inference(sv,un,yc); t["bayes"]=time.time()-ts

        feats={}
        for i,name in enumerate(TONGUE_FEATURES):
            if name=="tongue_body": continue
            s=float(sv[i]); u=float(un[i]); yci=yd.get(name,0.0); cnt=yc.get(name,0)
            cnt_sev=count_to_severity(cnt) if name in SMALL_LESION_FEATS else 0.0
            combined=max(s,yci*0.85,cnt_sev)
            if combined>0.12 or cnt>0:
                feats[name]={"severity":round(combined,3),
                    "level":"high" if combined>0.60 else "moderate" if combined>0.35 else "mild",
                    "confidence":round(float(np.exp(-u*3)),2),"yolo_count":cnt}

        sdef=sorted(enumerate(tp),key=lambda x:-x[1])
        defs={T_DEFS[i]:{"probability":round(float(p),4),
               "probability_pct":f"{p*100:.1f}%","priority_rank":r,
               "foods":T_FOOD.get(T_DEFS[i],[]),
               "confidence_band":"high" if p>0.20 else "moderate" if p>0.10 else "low"}
              for r,(i,p) in enumerate(sdef,1)}
        top_=[{"rank":r,"issue":T_DEFS[i],"probability":f"{tp[i]*100:.1f}%",
               "priority":"HIGH" if tp[i]>0.20 else "MODERATE" if tp[i]>0.10 else "LOW",
               "top_foods":T_FOOD.get(T_DEFS[i],[])[:3]}
              for r,(i,p) in enumerate(sdef[:5],1) if p>0.08]

        return JSONResponse({"status":"success","modality":"tongue",
            "features_detected":feats,"deficiency_analysis":defs,"top_insights":top_,
            "timing_ms":{k:round(v*1000,1) for k,v in t.items()},
            "tongue_crop_b64":enc(crop),"posterior":tp.tolist(),
            "disclaimer":"FaceFuel tongue analysis provides wellness awareness only."})
    except Exception as e:
        print(f"[ERROR] /analyze/tongue: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Tongue analysis failed: {e}")


@app.post("/analyze/combined")
async def analyze_combined(face: UploadFile=File(...), tongue: UploadFile=File(...)):
    face_img=await decode_img(face); tongue_img=await decode_img(tongue)
    try:
        from step10_inference import (preprocess_image, analyze_color_features,
            run_yolo, extract_dinov2_features, run_severity_mlp,
            bayesian_inference_v2, format_output)
        from eye_inference import run_eye_inference
        from tongue_inference import (detect_and_crop_tongue, run_tongue_yolo,
            extract_tongue_features, run_tongue_severity, tongue_bayesian_inference,
            TONGUE_FEATURES, DEFICIENCIES as T_DEFS, FOOD_RECS as T_FOOD,
            SMALL_LESION_FEATS, count_to_severity)

        fm   = get_face_pipeline()["models"]
        prep = preprocess_image(face_img, fm)
        if prep is None:
            return JSONResponse({"status":"no_face_detected"})

        cf=analyze_color_features(prep["raw_rgb"],prep["regions"])
        yd,yc=run_yolo(prep["aligned_bgr"],fm)
        fv=extract_dinov2_features(prep["regions"],fm)
        sv,un=run_severity_mlp(fv,fm)
        fp_=bayesian_inference_v2(sv,un,cf,yc)
        face_result=format_output(sv,un,yd,yc,fp_,cf,{})
        face_post=face_result.get("posterior",[0]*11)

        eye_result  = run_eye_inference(prep["aligned_bgr"],DEVICE)
        eye_post    = eye_result.get("posterior",[0]*22)
        eye_feats   = eye_result.get("features_detected",{})

        tm=get_tongue_pipeline()["models"]
        crop=detect_and_crop_tongue(tongue_img,tm)
        tyd,tyc=run_tongue_yolo(crop,tm)
        tfv=extract_tongue_features(crop,tm)
        tsv,tun=run_tongue_severity(tfv,tm)
        tpost=tongue_bayesian_inference(tsv,tun,tyc)
        tongue_post=tpost.tolist() if hasattr(tpost,"tolist") else list(tpost)

        tongue_feats={}
        for i,name in enumerate(TONGUE_FEATURES):
            if name=="tongue_body": continue
            s=float(tsv[i]); u=float(tun[i])
            yci=tyd.get(name,0.0); cnt=tyc.get(name,0)
            cnt_sev=count_to_severity(cnt) if name in SMALL_LESION_FEATS else 0.0
            combined=max(s,yci*0.85,cnt_sev)
            if combined>0.12 or cnt>0:
                tongue_feats[name]={"severity":round(combined,3),
                    "level":"high" if combined>0.60 else "moderate" if combined>0.35 else "mild",
                    "confidence":round(float(np.exp(-u*3)),2),"yolo_count":cnt}

        fused = fuse_posteriors(face_post, tongue_post, eye_post)
        out   = build_response(fused,
                               face_result.get("features_detected",{}),
                               tongue_feats, eye_feats, {})

        _,tbuf=cv2.imencode(".jpg",cv2.resize(crop,(300,300)),
                             [cv2.IMWRITE_JPEG_QUALITY,88])
        return JSONResponse({"status":"success","modality":"face+eye+tongue",
            "face_features":face_result.get("features_detected",{}),
            "eye_features":eye_feats,"tongue_features":tongue_feats,**out,
            "aligned_face_b64":enc(prep["aligned_bgr"]),
            "tongue_crop_b64":base64.b64encode(tbuf).decode(),
            "disclaimer":
                "FaceFuel v4 provides wellness awareness only — not medical diagnosis. "
                "Three modalities: face skin, eyes, tongue. 22 health dimensions assessed."})
    except Exception as e:
        print(f"[ERROR] /analyze/combined: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, f"Combined analysis failed: {e}")


if __name__ == "__main__":
    print(f"\nStarting FaceFuel v4 → http://localhost:{PORT}")
    uvicorn.run("server_v4:app", host=HOST, port=PORT,
                reload=False, workers=1, log_level="info")