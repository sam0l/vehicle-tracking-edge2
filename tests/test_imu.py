import pytest
import logging
from src.imu import IMU
from unittest.mock import Mock, patch

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture
def imu_config():
    return {
        "i2c_bus": 4,
        "i2c_addresses": ["0x68", "0x69"],
        "sample_rate": 100,
        "accel_range": 2,
        "gyro_range": 250
    }

def test_imu_initialize_success(imu_config, mocker):
    mock_bus = Mock(write_byte_data=Mock(), read_word_data=Mock(return_value=0))
    mocker.patch("smbus.SMBus", return_value=mock_bus)
    imu = IMU(**imu_config)
    logger.debug(f"Testing IMU initialization with address {imu_config['i2c_addresses']}")
    assert imu.initialize(), "IMU initialization failed"
    assert imu.address == 0x68, "IMU address mismatch"
    logger.debug("IMU initialized successfully")

def test_imu_initialize_failure(imu_config, mocker):
    mocker.patch("smbus.SMBus", side_effect=Exception("I2C error"))
    imu = IMU(**imu_config)
    logger.debug("Testing IMU initialization failure")
    assert not imu.initialize(), "IMU initialization should have failed"
    logger.debug("IMU initialization failed as expected")

def test_imu_read_data(imu_config, mocker):
    mock_bus = Mock(
        write_byte_data=Mock(),
        read_word_data=Mock(side_effect=[16384, 0, 0, 131, 0, 0])  # accel_x=1g, others 0
    )
    mocker.patch("smbus.SMBus", return_value=mock_bus)
    imu = IMU(**imu_config)
    imu.initialize()
    logger.debug("Testing IMU data reading")
    data = imu.read_data()
    assert data is not None, "IMU data should not be None"
    assert len(data["acceleration"]) == 3, "Acceleration data length mismatch"
    assert abs(data["acceleration"][0] - 1.0) < 0.01, f"Acceleration mismatch: {data['acceleration']}"
    assert data["is_stationary"] is True, "Stationary detection incorrect"
    logger.debug(f"IMU data: {data}")

def test_imu_update_gps(imu_config, mocker):
    mock_bus = Mock(write_byte_data=Mock(), read_word_data=Mock(return_value=0))
    mocker.patch("smbus.SMBus", return_value=mock_bus)
    imu = IMU(**imu_config)
    imu.initialize()
    gps_data = {"latitude": 12.34, "longitude": 56.78}
    logger.debug("Testing IMU GPS update")
    imu.update_gps(gps_data)
    assert imu.last_position == [12.34, 56.78], f"Position mismatch: {imu.last_position}"
    logger.debug("IMU GPS updated successfully")

def test_imu_close(imu_config, mocker):
    mock_bus = Mock(write_byte_data=Mock(), read_word_data=Mock(return_value=0), close=Mock())
    mocker.patch("smbus.SMBus", return_value=mock_bus)
    imu = IMU(**imu_config)
    imu.initialize()
    logger.debug("Testing IMU close")
    imu.close()
    mock_bus.close.assert_called_once()
    assert imu.bus is None, "IMU bus should be None after close"
    logger.debug("IMU closed successfully")