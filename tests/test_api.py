import pytest
from fastapi.testclient import TestClient
from src.main import app
from pathlib import Path

client = TestClient(app)
TEST_IMAGE_PATH = "gun1.jpg"

@pytest.fixture
def sample_image():
    
    with open(TEST_IMAGE_PATH, "rb") as image_file:
        yield image_file

def post_image(endpoint: str, sample_image, data: dict = None):
    
    files = {"file": ("sample_image.jpg", sample_image, "image/jpeg")}
    return client.post(endpoint, files=files, data=data or {})

def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    response_json = response.json()
    
    assert all(key in response_json for key in ["model_name", "gun_detector_model", "semantic_segmentation_model"])

@pytest.mark.parametrize("endpoint, expected_keys", [
    ("/detect_guns", ["n_detections", "boxes", "labels"]),
    ("/detect_people", ["n_detections", "polygons", "boxes", "labels"]),
    ("/detect", ["detection", "segmentation"]),
    ("/guns", []),
    ("/people", []),
])
def test_detection_endpoints(endpoint, expected_keys, sample_image):
    response = post_image(endpoint, sample_image, {"threshold": "0.5"})
    assert response.status_code == 200
    json_response = response.json()
    
    if expected_keys:
        assert all(key in json_response for key in expected_keys)

    validate_detections(endpoint, json_response)

def validate_detections(endpoint, json_response):
    
    if endpoint == "/guns":
        for gun in json_response:
            assert all(key in gun for key in ["gun_type", "location"])
            assert "x" in gun["location"]
            assert "y" in gun["location"]

    if endpoint == "/people":
        for person in json_response:
            assert all(key in person for key in ["person_type", "location", "area"])
            assert "x" in person["location"]
            assert "y" in person["location"]

@pytest.mark.parametrize("endpoint, data", [
    ("/annotate_guns", {"threshold": "0.5"}),
    ("/annotate_people", {"threshold": "0.5", "draw_boxes": "true"}),
    ("/annotate", {"threshold": "0.5"})
])
def test_annotation_endpoints(endpoint, data, sample_image):
    response = post_image(endpoint, sample_image, data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
