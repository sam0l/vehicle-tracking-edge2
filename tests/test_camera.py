import pytest
import logging
import numpy as np
from src.camera import Camera
from unittest.mock import Mock, patch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture
def camera_config():
    return {
        "device_id": "/dev/video1",
        "width": 640,
        "height": 360,
        "fps": 30
    }

def test_camera_initialize_success(camera_config, mocker):
    mock_cap = Mock(
        isOpened=Mock(return_value=True),
        read=Mock(return_value=(True, np.zeros((360, 640, 3), dtype=np.uint8))),
        set=Mock(return_value=True)
    )
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    camera = Camera(**camera_config)
    logger.debug(f"Testing camera initialization with device {camera_config['device_id']}")
    assert camera.initialize(), "Camera initialization failed"
    logger.debug("Camera initialized successfully")

def test_camera_initialize_failure(camera_config, mocker):
    mock_cap = Mock(isOpened=Mock(return_value=False))
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    camera = Camera(**camera_config)
    logger.debug("Testing camera initialization failure")
    assert not camera.initialize(), "Camera initialization should have failed"
    logger.debug("Camera initialization failed as expected")

def test_camera_get_frame(camera_config, mocker):
    mock_cap = Mock(
        isOpened=Mock(return_value=True),
        read=Mock(return_value=(True, np.zeros((360, 640, 3), dtype=np.uint8))),
        set=Mock(return_value=True)
    )
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    camera = Camera(**camera_config)
    camera.initialize()
    logger.debug("Testing camera frame capture")
    frame = camera.get_frame()
    assert frame is not None, "Frame should not be None"
    assert frame.shape == (360, 640, 3), f"Frame shape mismatch: {frame.shape}"
    logger.debug("Frame captured successfully")

def test_camera_get_frame_failure(camera_config, mocker):
    mock_cap = Mock(
        isOpened=Mock(return_value=True),
        read=Mock(return_value=(False, None)),
        set=Mock(return_value=True)
    )
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    camera = Camera(**camera_config)
    camera.initialize()
    logger.debug("Testing camera frame capture failure")
    frame = camera.get_frame()
    assert frame is None, "Frame should be None on failure"
    logger.debug("Frame capture failed as expected")

def test_camera_close(camera_config, mocker):
    mock_cap = Mock(
        isOpened=Mock(return_value=True),
        read=Mock(return_value=(True, np.zeros((360, 640, 3), dtype=np.uint8))),
        set=Mock(return_value=True),
        release=Mock()
    )
    mocker.patch("cv2.VideoCapture", return_value=mock_cap)
    camera = Camera(**camera_config)
    camera.initialize()
    logger.debug("Testing camera close")
    camera.close()
    mock_cap.release.assert_called_once()
    assert camera.cap is None, "Camera cap should be None after close"
    logger.debug("Camera closed successfully")