"""
API smoke tests. Need the model weights (skipped automatically otherwise) and
load every model, so they take ~15 s. Uses synthetic images only.
"""
import io

import numpy as np
import pytest
from PIL import Image

from facefuel import paths

missing = [k for k in paths.KEYS if not any(p.is_file() for p in paths.candidates(k))]
pytestmark = pytest.mark.skipif(bool(missing), reason=f"model weights missing: {missing}")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import server
    with TestClient(server.app) as c:
        yield c


def png(color=(200, 170, 150), size=(400, 400)):
    buf = io.BytesIO()
    Image.fromarray(np.full((*size, 3), color, dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_health_reports_every_modality_ready(client):
    h = client.get("/health").json()
    assert h["status"] == "healthy"
    assert set(h["modalities"].values()) == {"ready"}


def test_info_lists_inactive_classes(client):
    info = client.get("/api/info").json()
    assert "acne" in info["modalities"]["face"]["inactive_classes"]
    assert "acne" not in info["modalities"]["face"]["active_classes"]


def test_blank_image_is_no_face(client):
    r = client.post("/analyze", files={"file": ("blank.png", png(), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "no_face_detected" and "disclaimer" in j


def test_tongue_endpoint_schema_and_warning(client):
    r = client.post("/analyze/tongue", files={"file": ("t.png", png((150, 90, 160)), "image/png")})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "success" and j["modalities_run"] == ["tongue"]
    assert j["disclaimer"]
    if not j["tongue_localized"]:
        assert j["warnings"]
    for v in j["deficiency_analysis"].values():
        assert v["status"] in {"flagged", "no_signal", "not_assessed"}
        assert "face" not in v["sources"] and "eye" not in v["sources"]


def test_rejects_non_image(client):
    r = client.post("/analyze/tongue", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
