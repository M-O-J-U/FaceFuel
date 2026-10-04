"""
FaceFuel — tri-modal (face + eye + tongue) visual wellness screening.

Runtime package used by server.py. Layout:
  paths.py   — every weight/model file location, resolved in one place
  schema.py  — class lists, deficiency framework, feature→deficiency maps, advice
  models.py  — shared DINOv2, severity-MLP loading (v4 checkpoint format), YOLO cache
  face.py    — MediaPipe alignment + face YOLO + face MLP + colour analysis
  eye.py     — eye YOLO + eye MLP (scleral-icterus false-positive guards)
  tongue.py  — tongue localisation + tongue YOLO + tongue MLP
  fusion.py  — per-modality evidence scoring, product-of-experts fusion, response building

FaceFuel is a wellness-awareness research prototype, NOT a medical device.
"""

__version__ = "4.1.0"

DISCLAIMER = (
    "FaceFuel is a research prototype for wellness awareness only. It is not a "
    "medical device and does not diagnose, treat or rule out any condition. Visual "
    "signs have many possible causes. Consult a qualified healthcare professional "
    "before making any health decision."
)
