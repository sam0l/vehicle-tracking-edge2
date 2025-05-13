import pytest
import logging
import yaml
import numpy as np
from src.main import VehicleTracker
from unittest.mock import Mock, patch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture
def tracker_config(tmp_path):
    config = {
        "gps": {"port": "/dev/ttyUSB1", "baudrate": 115200, "timeout": 1, "power_delay": 2, "agps_delay": 5},
        "imu": {"i2c_bus": 4, "i2c_addresses": ["0x68"], "sample_rate": 100, "accel_range": 2, "gyro_range": 250},
        "camera": {"device_id": "/dev/video1", "width": 640, "height": 360, "fps": 30},
        "yolo": {"model_type": "onnx", "onnx_model_path": "models/yolov8n.onnx", "confidence_threshold": 0.7, "imgsz": 320, "iou_threshold": 0.45, "class_names": ["Stop"], "draw_boxes": True, "intra_op_num_threads": 2},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "file": "vehicle_tracker.log", "offline_file": "offline_data.json", "interval": {"gps": 1, "imu": 0.05, "camera": 0.5}},
        "detection": {"deduplication_timeout": 10, "distance_threshold": 0.001},
        "backend": {"url": "http://localhost", "endpoint_prefix": "/api", "telemetry_endpoint": "/telemetry", "detection_endpoint": "/detections", "sim_data_endpoint": "/sim-data", "database_url": "postgresql://user:pass@localhost/db"},
        "api": {"port": 5000}
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)

def test_vehicle_tracker_initialize_success(tracker_config, mocker):
    mocker.patch("src.gps.GPS.initialize", return_value=True)
    mocker.patch("src.imu.IMU.initialize", return_value=True)
    mocker.patch("src.camera.Camera.initialize", return_value=True)
    mocker.patch("src.sign_detection.SignDetector.__init__", return_value=None)
    tracker = VehicleTracker(tracker_config)
    logger.debug("Testing VehicleTracker initialization")
    assert tracker.initialize(), "VehicleTracker initialization failed"
    assert tracker.camera_initialized, "Camera should be initialized"
    logger.debug("VehicleTracker initialized successfully")

def test_vehicle_tracker_calculate_speed(tracker_config, mocker):
    mocker.patch("src.gps.GPS.initialize", return_value=True)
    mocker.patch("src.imu.IMU.initialize", return_value=True)
    mocker.patch("src.camera.Camera.initialize", return_value=True)
    mocker.patch("src.sign_detection.SignDetector.__init__", return_value=None)
    tracker = VehicleTracker(tracker_config)
    gps_data = {"speed": 10.0, "satellites": 6}
    imu_data = {"speed": 9.0}
    logger.debug("Testing speed calculation")
    speed = tracker.calculate_speed(gps_data, imu_data)
    assert abs(speed - 9.8) < 0.01, f"Speed mismatch: {speed}"
    logger.debug(f"Calculated speed: {speed}")

def test_vehicle_tracker_filter_duplicate_detections(tracker_config, mocker):
    mocker.patch("src.gps.GPS.initialize", return_value=True)
    mocker.patch("src.imu.IMU.initialize", return_value=True)
    mocker.patch("src.camera.Camera.initialize", return_value=True)
    mocker.patch("src.sign_detection.SignDetector.__init__", return_value=None)
    tracker = VehicleTracker(tracker_config)
    signs = [{"label": "Stop", "confidence": 0.8, "box": [100, 100, 200, 200]}]
    position = (12.34, 56.78)
    current_time = 1000.0
    tracker.recent_detections["Stop"] = {"time": 990.0, "position": (12.34, 56.78)}
    logger.debug("Testing duplicate detection filtering")
    filtered = tracker.filter_duplicate_detections(signs, position, current_time)
    assert len(filtered) == 0, "Duplicate detection should be filtered"
    logger.debug("Duplicate detection filtered successfully")

def test_vehicle_tracker_send_data_offline(tracker_config, mocker):
    mocker.patch("src.gps.GPS.initialize", return_value=True)
    mocker.patch("src.imu.IMU.initialize", return_value=True)
    mocker.patch("src.camera.Camera.initialize", return_value=True)
    mocker.patch("src.sign_detection.SignDetector.__init__", return_value=None)
    mocker.patch("src.main.VehicleTracker.check_connectivity", return_value=False)
    tracker = VehicleTracker(tracker_config)
    data = {"timestamp": "2025-05-13 19:30:00", "gps": {"latitude": 12.34, "longitude": 56.78}}
    logger.debug("Testing offline data logging")
    tracker.send_data(data)
    assert len(tracker.offline_data) == 1, "Offline data should be logged"
    logger.debug(f"Offline data logged: {tracker.offline_data}")