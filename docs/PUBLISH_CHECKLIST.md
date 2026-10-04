# GitHub Publish Checklist

Prepared October 2026. `.gitignore` enforces most of this automatically; the list below
spells out the decisions behind it.

## ⚠️ Before anything else: the public repo already contains personal photos

`github.com/M-O-J-U/FaceFuel` is **public**, and its only commit (`b5b21eb`, "FaceFuel v3")
contains files that should not be public:

- `Myself.jpeg`, `Myself2.jpeg`, `facefuel_outputs/Myself_result.jpg/.json` — personal photos
- `test/*.jpg|webp` — 15 photos of people downloaded from the web (third-party copyright)
- `Acne-Treatment-1-1024x724.jpg`, `healthy.avif`, `healthy2.avif` — third-party images
- `yolo11m.pt`, `yolo26n.pt`, `yolov8m.pt` (~98 MB), `__pycache__/` — unnecessary binaries

Deleting them in a new commit is **not enough**: they stay in history and are
downloadable via the old commit. Because the repo has a single commit, the clean options are:

| Option | What happens | Thoroughness |
|---|---|---|
| **A. Delete and recreate the GitHub repo** (recommended) | Settings → Delete repository; create an empty `FaceFuel` repo; push a fresh history | Complete. Old commit URLs stop resolving. Stars and issues are lost (likely few). |
| B. Force-push a new root commit | `git checkout --orphan clean && git commit && git push -f origin clean:main` | The old commit stays reachable by SHA until GitHub garbage-collects it; ask GitHub Support to purge cached views. |

Both are destructive and need **your** go-ahead. Nothing has been pushed or force-pushed.
Whichever you choose, the working tree is already cleaned: those files now live in
`local/` (git-ignored) and show as deletions in `git status`.

## Ships (tracked)

| Path | Why |
|---|---|
| `server.py`, `facefuel/`, `static/index.html` | The application |
| `pipeline/`, `scripts/` | Reproducible training pipeline and tools |
| `tests/` | Regression tests |
| `requirements*.txt`, `Dockerfile`, `.dockerignore`, `.gitignore` | Setup and deployment |
| `README.md`, `CLAUDE.md`, `docs/*.md` | Documentation (project summary, research notes, deployment, this checklist) |
| `docs/figures/*.png` | Pipeline, fusion, eye and tongue diagrams, plus the UI screenshot (contains no faces) |
| `paper_results/paper */*.tex, *.bib, *.png`, `paper_results/generators/` | Paper sources and diagrams. PDFs are excluded (they are public on Zenodo and may contain sample photos) |
| `legacy/` | v1–v3 code, kept so Papers 1–3 remain reproducible; the server does not use it |

## Stays local (git-ignored)

| Path | Size | Why |
|---|---|---|
| `facefuel_datasets*/`, `tongue_datasets*/`, `eye_datasets*/` | many GB | Raw third-party datasets; licences generally forbid redistribution |
| `facefuel_{face,tongue,eye}_v4/`, `facefuel_features/` | GB | Derived from the above |
| `runs/`, `training_runs/`, `facefuel_models/`, `weights/`, `*.pt`, `*.task` | ~300 MB | Distribute the 7 inference files as a **Release asset** instead (below) |
| `local/` | ~250 MB | Personal photos, test photos, promo video (140 MB — over GitHub's 100 MB file limit), `facefuel_structure.txt` scan dump, `.bak` backups, base YOLO weights (re-downloadable), old frontend copies, logos/screenshots in `facefuel extras/` |
| `paper_results/**/cover letter.txt`, `pp_statement.txt`, `*.zip` | small | Journal-submission correspondence and duplicate archives |
| `__pycache__/`, `.vscode/`, `.playwright-mcp/` | — | Tooling |

## Model weights as a GitHub Release

```bash
python scripts/collect_weights.py                # → weights/ + MANIFEST.txt (sha256)
# zip weights/ as facefuel-weights-v4.zip and attach it to a GitHub Release (assets up to 2 GB)
```

**Licence check first.** The weights are derived from third-party datasets (Roboflow,
Kaggle, Mendeley, IEEE DataPort, …). Several are CC BY-NC or "research use only". List each
source and its licence in the release notes; if any forbids derivative distribution,
publish the weights on request only.

## Decisions only you can make

1. **History cleanup:** Option A or B above.
2. ~~Code licence~~: **decided — PolyForm Strict 1.0.0** (noncommercial use only, no
   redistribution, no modified versions). Note that GitHub's Terms of Service still let
   any user view and fork a *public* repository within GitHub; the licence governs what
   they may legally do with it. For maximum protection, keep the repo private and share
   access individually.
3. **Weights:** public release, on request only, or not at all (depends on the licence check above).
4. ~~Promo video~~: already on YouTube (https://youtu.be/_RNtt8QNQOs) and linked from the README.
5. **Logos:** `local/facefuel extras/logo*.png` could replace the generated three-light mark if you prefer your own branding.

## Final pre-push checks

```bash
python -m pytest tests -q                                  # all green
git status --short | grep -v '^??'                         # review every staged change
git ls-files | grep -Ei '\.(pt|jpe?g|avif|webp|mp4|npz)$'  # should print only docs/figures or paper images
git ls-files | xargs -I{} du -k "{}" | sort -n | tail      # no file > a few MB
```
