"""FaceFuel v3 — Tri-Modal Server (Face + Eye + Tongue)
POST /analyze           ← selfie → face + eye (16 deficiencies)
POST /analyze/tongue    ← tongue photo only
POST /analyze/combined  ← selfie + tongue → all 3 modalities
GET  /health  |  GET  /
"""
import os,sys,time,base64,logging,traceback
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
import cv2
import torch
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import HTMLResponse,JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

os.environ["GLOG_minloglevel"]="2"
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
sys.path.insert(0,str(Path(__file__).parent))

HOST="0.0.0.0"; PORT=8000; MAX_MB=15
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

_face_pipeline=None; _tongue_pipeline=None; _eye_pipeline=None

def get_face_pipeline():
    global _face_pipeline
    if _face_pipeline is None:
        from step10_inference import get_models
        _face_pipeline={"models":get_models(DEVICE)}
    return _face_pipeline

def get_tongue_pipeline():
    global _tongue_pipeline
    if _tongue_pipeline is None:
        from Phase7_tongue_inference import get_tongue_models
        _tongue_pipeline={"models":get_tongue_models(DEVICE)}
    return _tongue_pipeline

def get_eye_pipeline():
    global _eye_pipeline
    if _eye_pipeline is None:
        from eye_inference import get_eye_models
        _eye_pipeline={"models":get_eye_models(DEVICE)}
    return _eye_pipeline

FACE_DEFS=["iron_deficiency","b12_deficiency","vitamin_d_deficiency","zinc_deficiency",
           "omega3_deficiency","vitamin_a_deficiency","vitamin_c_deficiency",
           "poor_sleep_quality","hormonal_imbalance","dehydration","high_stress"]
TONGUE_DEFS=FACE_DEFS+["liver_stress","gut_dysbiosis","hypothyroid","folate_deficiency"]
EYE_DEFS=TONGUE_DEFS+["cholesterol_imbalance"]
ALL_DEFS=list(dict.fromkeys(EYE_DEFS))

FOOD_RECS={
    "iron_deficiency":["spinach","lentils","red meat","tofu","pumpkin seeds"],
    "b12_deficiency":["eggs","dairy","salmon","beef liver","fortified cereals"],
    "vitamin_d_deficiency":["fatty fish","egg yolks","fortified milk","mushrooms"],
    "zinc_deficiency":["oysters","beef","chickpeas","cashews","pumpkin seeds"],
    "omega3_deficiency":["salmon","walnuts","flaxseed","chia seeds","mackerel"],
    "vitamin_a_deficiency":["sweet potato","carrots","kale","egg yolks","liver"],
    "vitamin_c_deficiency":["citrus fruits","bell peppers","broccoli","kiwi"],
    "poor_sleep_quality":["improve sleep schedule","reduce caffeine after 2pm","magnesium"],
    "hormonal_imbalance":["see a doctor","reduce sugar","healthy fats","fiber"],
    "dehydration":["drink 8+ glasses water daily","cucumber","watermelon"],
    "high_stress":["meditation","exercise","B-complex vitamins","magnesium"],
    "liver_stress":["reduce alcohol","leafy greens","beets","milk thistle tea"],
    "gut_dysbiosis":["probiotics","fermented foods","fiber","reduce sugar"],
    "hypothyroid":["consult doctor","iodine-rich foods","selenium","zinc"],
    "folate_deficiency":["leafy greens","lentils","asparagus","fortified cereals"],
    "cholesterol_imbalance":["oats","beans","avocado","olive oil","salmon"],
}
ADVICE={
    "iron_deficiency":"Pair with Vitamin C to boost absorption.",
    "b12_deficiency":"Mainly from animal sources. Vegans must supplement.",
    "vitamin_d_deficiency":"15–30 min sunlight daily. Consider D3 in winter.",
    "zinc_deficiency":"Soak legumes to reduce phytates.",
    "omega3_deficiency":"Aim for 2 servings fatty fish per week.",
    "vitamin_a_deficiency":"Fat-soluble — pair with healthy fats.",
    "vitamin_c_deficiency":"Eat some raw fruits/vegetables daily.",
    "poor_sleep_quality":"Consistent sleep/wake times. Aim 7–9 hrs.",
    "hormonal_imbalance":"Requires medical evaluation.",
    "dehydration":"Aim for pale yellow urine throughout the day.",
    "high_stress":"Both diet and stress management needed.",
    "liver_stress":"Reduce processed foods and alcohol.",
    "gut_dysbiosis":"Increase fiber and fermented foods.",
    "hypothyroid":"Consult doctor for thyroid testing.",
    "folate_deficiency":"Especially important during pregnancy.",
    "cholesterol_imbalance":"Reduce saturated fats; increase soluble fiber and omega-3.",
}


def extract_face_posterior(face_result):
    da=face_result.get("deficiency_analysis",{})
    if not da: return [1.0/len(FACE_DEFS)]*len(FACE_DEFS)
    post=[]
    for name in FACE_DEFS:
        entry=da.get(name,{})
        p=entry.get("probability",0.0)
        if isinstance(p,str): p=float(p.replace("%",""))/100.0
        post.append(float(p))
    total=sum(post)+1e-9
    return [p/total for p in post]


def fuse_all_posteriors(face_post,tongue_post,eye_post,
                        eye_has_findings=True,tongue_has_findings=True):
    fw,tw,ew=0.40,0.35,0.25
    if not eye_has_findings:    ew=0.0
    if not tongue_has_findings: tw=0.0
    total_w=fw+tw+ew
    if total_w>0: fw/=total_w; tw/=total_w; ew/=total_w

    def get(post,defs,name):
        return post[defs.index(name)] if name in defs and len(post)>defs.index(name) else 0.0

    log_fused={}
    for name in ALL_DEFS:
        fp=get(face_post,FACE_DEFS,name) if fw>0 else 0.0
        tp=get(tongue_post,TONGUE_DEFS,name) if tw>0 else 0.0
        ep=get(eye_post,EYE_DEFS,name) if ew>0 else 0.0
        log_val=active_w=0.0
        if fp>0 and fw>0: log_val+=fw*np.log(fp+1e-9); active_w+=fw
        if tp>0 and tw>0: log_val+=tw*np.log(tp+1e-9); active_w+=tw
        if ep>0 and ew>0: log_val+=ew*np.log(ep+1e-9); active_w+=ew
        if active_w>0:
            log_fused[name]=log_val*(0.85 if active_w<0.8 else 1.0)
        else:
            log_fused[name]=np.log(1.0/len(ALL_DEFS))

    vals=np.array(list(log_fused.values()),dtype=np.float32)
    vals=vals-vals.max()
    probs=np.exp(vals); probs=probs/(probs.sum()+1e-9)
    return {name:float(p) for name,p in zip(log_fused.keys(),probs)}


def build_output(fused,face_feats,tongue_feats,eye_feats,timing):
    sdef=sorted(fused.items(),key=lambda x:-x[1])
    tongue_excl={"liver_stress","gut_dysbiosis","hypothyroid","folate_deficiency"}
    eye_excl={"cholesterol_imbalance"}

    # Only attribute modality as source if it actually produced detections
    has_tongue=bool(tongue_feats and any(
        isinstance(v,dict) and v.get("severity",0)>0 for v in tongue_feats.values()))
    has_eye=bool(eye_feats and any(
        isinstance(v,dict) and v.get("severity",0)>0.10 for v in eye_feats.values()))

    def src(name):
        parts=[]
        if name in FACE_DEFS: parts.append("face")
        if name in tongue_excl and has_tongue: parts.append("tongue")
        if name in eye_excl    and has_eye:    parts.append("eye")
        if not parts: parts=["face"]
        return "+".join(parts)

    da={name:{"probability":round(p,4),"probability_pct":f"{p*100:.1f}%",
               "priority_rank":r,"foods":FOOD_RECS.get(name,[]),
               "advice":ADVICE.get(name,""),
               "confidence_band":"high" if p>0.20 else "moderate" if p>0.10 else "low",
               "source":src(name)}
        for r,(name,p) in enumerate(sdef,1)}
    top=[{"rank":r,"issue":name,"probability":f"{p*100:.1f}%",
           "priority":"HIGH" if p>0.20 else "MODERATE" if p>0.10 else "LOW",
           "top_foods":FOOD_RECS.get(name,[])[:3],
           "advice":ADVICE.get(name,""),"source":src(name)}
          for r,(name,p) in enumerate(sdef[:5],1) if p>0.08]
    return {"deficiency_analysis":da,"top_insights":top,
            "timing_ms":{k:round(v*1000,1) if isinstance(v,float) else v
                         for k,v in timing.items()}}


async def decode_img(file):
    data=await file.read()
    if len(data)>MAX_MB*1024*1024: raise HTTPException(413,f"Max {MAX_MB}MB.")
    img=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,"Could not decode image.")
    return img

def enc(bgr,size=(300,300)):
    _,buf=cv2.imencode(".jpg",cv2.resize(bgr,size),[cv2.IMWRITE_JPEG_QUALITY,88])
    return base64.b64encode(buf).decode()


@asynccontextmanager
async def lifespan(app):
    print(f"\n{'='*58}\n  FaceFuel v3 — Tri-Modal (Face+Eye+Tongue)")
    print(f"  Device: {DEVICE}"+
          (f"  GPU: {torch.cuda.get_device_name(0)}" if DEVICE=="cuda" else ""))
    print(f"  16 deficiency dimensions  |  3 modalities\n{'='*58}")
    for name,loader in [("Face+Eye",get_face_pipeline),
                         ("Tongue",get_tongue_pipeline),
                         ("Eye",get_eye_pipeline)]:
        try: loader(); print(f"  ✅ {name} ready")
        except Exception as e: print(f"  ⚠ {name}: {e}")
    print("  Server ready.\n")
    yield

app=FastAPI(title="FaceFuel v3",version="3.0.0",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],
                   allow_methods=["GET","POST"],allow_headers=["*"])
STATIC_DIR=Path(__file__).parent/"static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static",StaticFiles(directory=str(STATIC_DIR)),name="static")


@app.get("/",response_class=HTMLResponse)
async def frontend():
    idx=STATIC_DIR/"index.html"
    return HTMLResponse(idx.read_text(encoding="utf-8") if idx.exists()
                        else "<h1>FaceFuel v3</h1>")

@app.get("/health")
async def health():
    gpu={"name":torch.cuda.get_device_name(0)} if DEVICE=="cuda" else {}
    return {"status":"healthy","version":"3.0.0","device":DEVICE,"gpu":gpu,
            "modalities":3,"deficiencies":16,
            "face_loaded":_face_pipeline is not None,
            "tongue_loaded":_tongue_pipeline is not None,
            "eye_loaded":_eye_pipeline is not None}


@app.post("/analyze")
async def analyze_face(file:UploadFile=File(...)):
    img=await decode_img(file)
    try:
        from step10_inference import (preprocess_image,analyze_color_features,run_yolo,
            extract_dinov2_features,run_severity_mlp,bayesian_inference_v2,format_output)
        from eye_inference import run_eye_inference
        fm=get_face_pipeline()["models"]; t={}
        ts=time.time(); prep=preprocess_image(img,fm); t["preprocess"]=time.time()-ts
        if prep is None:
            return JSONResponse({"status":"no_face_detected",
                "message":"No face detected. Use a clear front-facing photo."})
        ts=time.time(); cf=analyze_color_features(prep["raw_rgb"],prep["regions"]); t["color"]=time.time()-ts
        ts=time.time(); yd,yc=run_yolo(prep["aligned_bgr"],fm); t["yolo_face"]=time.time()-ts
        ts=time.time(); fv=extract_dinov2_features(prep["regions"],fm); t["dinov2"]=time.time()-ts
        ts=time.time(); sv,un=run_severity_mlp(fv,fm); t["severity"]=time.time()-ts
        ts=time.time(); fp=bayesian_inference_v2(sv,un,cf,yc); t["bayes_face"]=time.time()-ts
        face_result=format_output(sv,un,yd,yc,fp,cf,t)
        ts=time.time(); eye_result=run_eye_inference(prep["aligned_bgr"],DEVICE); t["eye"]=time.time()-ts
        fused=fuse_all_posteriors(extract_face_posterior(face_result),[0.0]*15,
                                   eye_result.get("posterior",[0]*16),
                                   eye_has_findings=eye_result.get("has_findings",False),
                                   tongue_has_findings=False)
        out=build_output(fused,face_result.get("features_detected",{}),
                          {},eye_result.get("features_detected",{}),t)
        return JSONResponse({"status":"success","modality":"face+eye",
            "face_features":face_result.get("features_detected",{}),
            "eye_features":eye_result.get("features_detected",{}),"tongue_features":{},
            **out,"aligned_face_b64":enc(prep["aligned_bgr"]),
            "disclaimer":"FaceFuel v3 provides wellness awareness only — not medical diagnosis."})
    except Exception as e:
        print(f"[ERROR] /analyze: {e}\n{traceback.format_exc()}")
        raise HTTPException(500,f"Analysis failed: {e}")


@app.post("/analyze/tongue")
async def analyze_tongue(file:UploadFile=File(...)):
    img=await decode_img(file)
    try:
        from Phase7_tongue_inference import (detect_and_crop_tongue,run_tongue_yolo,
            extract_tongue_features,run_tongue_severity,tongue_bayesian_inference,
            TONGUE_FEATURES,DEFICIENCIES as T_DEFS,FOOD_RECS as T_FOOD,
            SMALL_LESION_FEATS,count_to_severity)
        m=get_tongue_pipeline()["models"]; t={}
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
                srcs=[]
                if s>0.12: srcs.append("dinov2")
                if yci>0:  srcs.append(f"yolo(×{cnt})" if cnt>1 else "yolo")
                feats[name]={"severity":round(combined,3),
                    "level":"high" if combined>0.60 else "moderate" if combined>0.35 else "mild",
                    "confidence":round(float(np.exp(-u*3)),2),"yolo_count":cnt,"detected_by":srcs}
        sdef=sorted(enumerate(tp),key=lambda x:-x[1])
        defs={T_DEFS[i]:{"probability":round(float(p),4),"probability_pct":f"{p*100:.1f}%",
               "priority_rank":r,"foods":T_FOOD.get(T_DEFS[i],[]),
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
        raise HTTPException(500,f"Tongue analysis failed: {e}")


@app.post("/analyze/combined")
async def analyze_combined(face:UploadFile=File(...),tongue:UploadFile=File(...)):
    face_img=await decode_img(face); tongue_img=await decode_img(tongue)
    try:
        from step10_inference import (preprocess_image,analyze_color_features,run_yolo,
            extract_dinov2_features,run_severity_mlp,bayesian_inference_v2,format_output)
        from eye_inference import run_eye_inference
        from Phase7_tongue_inference import (detect_and_crop_tongue,run_tongue_yolo,
            extract_tongue_features,run_tongue_severity,tongue_bayesian_inference,
            TONGUE_FEATURES,DEFICIENCIES as T_DEFS,FOOD_RECS as T_FOOD,
            SMALL_LESION_FEATS,count_to_severity)
        fm=get_face_pipeline()["models"]
        prep=preprocess_image(face_img,fm)
        if prep is None:
            return JSONResponse({"status":"no_face_detected","message":"No face detected."})
        cf=analyze_color_features(prep["raw_rgb"],prep["regions"])
        yd,yc=run_yolo(prep["aligned_bgr"],fm)
        fv=extract_dinov2_features(prep["regions"],fm)
        sv,un=run_severity_mlp(fv,fm)
        fp_=bayesian_inference_v2(sv,un,cf,yc)
        face_result=format_output(sv,un,yd,yc,fp_,cf,{})
        face_post=extract_face_posterior(face_result)
        eye_result=run_eye_inference(prep["aligned_bgr"],DEVICE)
        eye_post=eye_result.get("posterior",[0]*16)
        eye_feats=eye_result.get("features_detected",{})
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
        fused=fuse_all_posteriors(face_post,tongue_post,eye_post,
                                  eye_has_findings=eye_result.get("has_findings",False),
                                  tongue_has_findings=bool(tongue_feats))
        out=build_output(fused,face_result.get("features_detected",{}),
                          tongue_feats,eye_feats,{})
        _,tbuf=cv2.imencode(".jpg",cv2.resize(crop,(300,300)),[cv2.IMWRITE_JPEG_QUALITY,88])
        return JSONResponse({"status":"success","modality":"face+eye+tongue",
            "face_features":face_result.get("features_detected",{}),"eye_features":eye_feats,
            "tongue_features":tongue_feats,**out,
            "aligned_face_b64":enc(prep["aligned_bgr"]),
            "tongue_crop_b64":base64.b64encode(tbuf).decode(),
            "disclaimer":"FaceFuel v3 — wellness awareness only. Not medical diagnosis."})
    except Exception as e:
        print(f"[ERROR] /analyze/combined: {e}\n{traceback.format_exc()}")
        raise HTTPException(500,f"Combined analysis failed: {e}")


if __name__=="__main__":
    print(f"\nStarting FaceFuel v3 → http://localhost:{PORT}")
    uvicorn.run("server:app",host=HOST,port=PORT,reload=False,workers=1,log_level="info")


# http://localhost:8000
