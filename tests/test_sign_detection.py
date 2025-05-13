```python
import pytest
import logging
import numpy as np
import yaml
from src.sign_detection import SignDetector
from unittest.mock import Mock, patch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture
def sign_detector_config(tmp_path):
    config = {
        "yolo": {
            "model_type": "onnx",
            "onnx_model_path": "models/yolov8n.onnx",
            "rknn_model_path": "models/yolov8n.rknn",
            "confidence_threshold": 0.7,
            "imgsz": 320,
            "iou_threshold": 0.45,
            "class_names": ["Stop", "Speed Limit 50"],
            "draw_boxes": True,
            "intra_op_num_threads": 2
        }
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)

def test_sign_detector_initialize_onnx(sign_detector_config, mocker):
    mocker.patch("onnxruntime.InferenceSession", return_value=Mock(run=Mock()))
    mocker.patch("pyopencl.get_platforms", return_value=[Mock(get_devices=Mock(return_value=[Mock()]))])
    detector = SignDetector(sign_detector_config)
    logger.debug("Testing SignDetector ONNX initialization")
    assert detector.model_type == "onnx", "Model type should be ONNX"
    assert detector.session is not None, "ONNX session should be initialized"
    logger.debug("SignDetector ONNX initialized successfully")

def test_sign_detector_initialize_rknn(sign_detector_config, mocker):
    mocker.patch("yaml.safe_load", return_value={"yolo": {"model_type": "rknn", "rknn_model_path": "models/yolov8n.rknn"}})
    mock_rknn = Mock(load_rknn=Mock(return_value=0), init_runtime=Mock(return_value=0), inference=Mock(return_value=[np.array([[100, 100, 200, 200, 0.8, 0]])]))
    mocker.patch("rknnlite.api.RKNNLite", return_value=mock_rknn)
    mocker.patch("pyopencl.get_platforms", return_value=[Mock(get_devices=Mock(return_value=[Mock()]))])
    detector = SignDetector(sign_detector_config)
    logger.debug("Testing SignDetector RKNNLite initialization")
    assert detector.model_type == "rknn", "Model type should be RKNN"
    assert detector.rknn is not None, "RKNNLite session should be initialized"
    logger.debug("SignDetector RKNNLite initialized successfully")

def test_sign_detector_detect_onnx(sign_detector_config, mocker):
    mock_session = Mock(run=Mock(return_value=[np.array([[100, 100, 200, 200, 0.8, 0]])]))
    mocker.patch("onnxruntime.InferenceSession", return_value=mock_session)
    mocker.patch("pyopencl.get_platforms", return_value=[Mock(get_devices=Mock(return_value=[Mock()]))])
    detector = SignDetector(sign_detector_config)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    logger.debug("Testing SignDetector ONNX detection")
    results = detector.detect(frame)
    assert len(results) == 1, f"Expected 1 detection, got {len(results)}"
    assert results[0]["label"] == "Stop", f"Label mismatch: {results[0]['label']}"
    assert abs(results[0]["confidence"] - 0.8) < 0.01, f"Confidence mismatch: {results[0]['confidence']}"
    logger.debug(f"Detection results: {results}")

def test_sign_detector_detect_rknn(sign_detector_config, mocker):
    mocker.patch("yaml.safe_load", return_value={"yolo": {"model_type": "rknn", "rknn_model_path": "models/yolov8n.rknn"}})
    mock_rknn = Mock(load_rknn=Mock(return_value=0), init_runtime=Mock(return_value=0), inference=Mock(return_value=[np.array([[100, 100, 200, 200, 0.8, 0]])]))
    mocker.patch("rknnlite.api.RKNNLite", return_value=mock_rknn)
    mocker.patch("pyopencl.get_platforms", return_value=[Mock(get_devices=Mock(return_value=[Mock()]))])
    detector = SignDetector(sign_detector_config)
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    logger.debug("Testing SignDetector RKNNLite detection")
    results = detector.detect(frame)
    assert len(results) == 1, f"Expected 1 detection, got {len(results)}"
    assert results[0]["label"] == "Stop", f"Label mismatch: {results[0]['label']}"
    assert abs(results[0]["confidence"] - 0.8) < 0.01, f"Confidence mismatch: {results[0]['confidence']}"
    logger.debug(f"Detection results: {results}")

def test_sign_detector_close_rknn(sign_detector_config, mocker):
    mocker.patch("yaml.safe_load", return_value={"yolo": {"model_type": "rknn", "rknn_model_path": "models/yolov8n.rknn"}})
    mock_rknn = Mock(load_rknn=Mock(return_value=0), init_runtime=Mock(return_value=0), release=Mock())
    mocker.patch("rknnlite.api.RKNNLite", return_value=mock_rknn)
    mocker.patch("pyopencl.get_platforms", return_value=[Mock(get_devices=Mock(return_value=[Mock()]))])
    detector = SignDetector(sign_detector_config)
    logger.debug("Testing SignDetector RKNNLite close")
    detector.close()
    mock_rknn.release.assert_called_once()
    logger.debug("SignDetector RKNNLite closed successfully")
```