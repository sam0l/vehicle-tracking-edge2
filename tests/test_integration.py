import pytest
import logging
import yaml
import numpy as np
from src.main import VehicleTracker
from src.gps import GPS
from src.imu import IMU
from src.camera import Camera
from src.sign_detection import SignDetector
from src.sim_monitor import SimMonitor
from unittest.mock import Mock, patch
from multiprocessing import Queue

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture
def integration_config(tmp_path):
    config = {
        "gps": {"port": "/dev/ttyUSB1", "baudrate": 115200, "timeout": 1, "power_delay": 2, "agps_delay": 5},
        "imu": {"i2c_bus": 4, "i2c_addresses": ["0x68"], "sample_rate": 100, "accel_range": 2, "gyro_range": 250},
        "camera": {"device_id": "/dev/video1", "width": 640, "height": 360, "fps": 30},
        "yolo": {"model_type": "onnx", "onnx_model_path": "models/yolov8n.onnx", "confidence_threshold": 0.7, "imgsz": 320, "iou_threshold": 0.45, "class_names": ["Stop"], "draw_boxes": True, "intra_op_num_threads": 2},
        "sim": {"port": "/dev/ttyUSB2", "baudrate": 115200, "check_interval": 60, "usage_file": "data_usage.json", "interfaces": ["ppp0"], "apn": "internet", "ussd_balance_code": "*221#", "modem_init_commands": ["AT+CFUN=1"], "initialization_retries": 3},
        "logging": {"level": "INFO", "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "file": "vehicle_tracker.log", "offline_file": "offline_data.json", "interval": {"gps": 1, "imu": 0.05, "camera": 0.5}},
        "detection": {"deduplication_timeout": 10, "distance_threshold": 0.001},
        "backend": {"url": "http://localhost", "endpoint_prefix": "/api", "telemetry_endpoint": "/telemetry", "detection_endpoint": "/detections", "sim_data_endpoint": "/sim-data", "database_url": "postgresql://user:pass@localhost/db"},
        "api": {"port": 5000}
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config, str(config_path)

def test_integration_data_flow(integration_config, mocker):
    config, config_path = integration_config
    # Mock hardware dependencies
    mocker.patch("serial.Serial", return_value=Mock(
        write=Mock(),
        read=Mock(return_value=b"+CGPSINFO: 1234.567890,N,09876.543210,E,10.0,500.0,6,,OK"),
        in_waiting=True
    ))
    mocker.patch("smbus.SMBus", return_value=Mock(
        write_byte_data=Mock(),
        read_word_data=Mock(side_effect=[16384, 0, 0, 131, 0, 0])
    ))
    mocker.patch("cv2.VideoCapture", return_value=Mock(
        isOpened=Mock(return_value=True),
        read=Mock(return_value=(True, np.zeros((360, 640, 3), dtype=np.uint8))),
        set=Mock(return_value=True)
    ))
    mocker.patch("onnxruntime.InferenceSession", return_value=Mock(run=Mock(return_value=[np.array([[100, 100, 200, 200, 0.8, 0]])])))
    mocker.patch("pyopencl.get_platforms", return_value=[Mock(get_devices=Mock(return_value=[Mock()]))])
    mocker.patch("aiohttp.ClientSession.post", return_value=Mock(__aenter__=Mock(return_value=Mock(status=200))))
    mocker.patch("sqlalchemy.create_engine", return_value=Mock())
    
    tracker = VehicleTracker(config_path)
    tracker.data_queue = Queue()
    
    logger.debug("Testing integration: Starting GPS process")
    tracker.gps_process()  # Simulate one cycle
    gps_data = tracker.data_queue.get_nowait()
    assert gps_data[0] == "gps", "Expected GPS data"
    assert abs(gps_data[1]["latitude"] - 12.576131) < 0.01, f"GPS latitude mismatch: {gps_data[1]['latitude']}"
    logger.debug(f"GPS data: {gps_data[1]}")
    
    logger.debug("Testing integration: Starting IMU process")
    tracker.imu_process()  # Simulate one cycle
    imu_data = tracker.data_queue.get_nowait()
    assert imu_data[0] == "imu", "Expected IMU data"
    assert len(imu_data[1]["acceleration"]) == 3, "IMU acceleration length mismatch"
    logger.debug(f"IMU data: {imu_data[1]}")
    
    logger.debug("Testing integration: Starting camera process")
    tracker.camera_initialized = True
    tracker.camera_process()  # Simulate one cycle
    camera_data = tracker.data_queue.get_nowait()
    assert camera_data[0] == "signs", "Expected signs data"
    assert len(camera_data[1]) == 1, f"Expected 1 detection, got {len(camera_data[1])}"
    logger.debug(f"Signs data: {camera_data[1]}")
    
    logger.debug("Testing integration: Starting transmission process")
    tracker.check_connectivity = Mock(return_value=True)
    tracker.data_transmission_process()  # Simulate one cycle
    assert tracker.batch, "Batch should contain data"
    logger.debug(f"Batch data: {tracker.batch}")
    
    logger.debug("Integration test completed successfully")