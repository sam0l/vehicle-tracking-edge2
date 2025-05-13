import pytest
import logging
from src.gps import GPS
from unittest.mock import Mock, patch

# Setup logging for tests
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture
def gps_config():
    return {
        "port": "/dev/ttyUSB1",
        "baudrate": 115200,
        "timeout": 1,
        "power_delay": 2,
        "agps_delay": 5
    }

def test_gps_initialize_success(gps_config, mocker):
    mocker.patch("serial.Serial", return_value=Mock(write=Mock(), read=Mock(return_value=b"OK")))
    gps = GPS(**gps_config)
    logger.debug(f"Testing GPS initialization with port {gps_config['port']}")
    assert gps.initialize(), "GPS initialization failed"
    logger.debug("GPS initialized successfully")

def test_gps_initialize_failure(gps_config, mocker):
    mocker.patch("serial.Serial", side_effect=Exception("Serial error"))
    gps = GPS(**gps_config)
    logger.debug(f"Testing GPS initialization failure with port {gps_config['port']}")
    assert not gps.initialize(), "GPS initialization should have failed"
    logger.debug("GPS initialization failed as expected")

def test_gps_get_data_valid(gps_config, mocker):
    mock_serial = Mock(
        write=Mock(),
        read=Mock(return_value=b"+CGPSINFO: 1234.567890,N,09876.543210,E,100.0,500.0,10,,OK")
    )
    mocker.patch("serial.Serial", return_value=mock_serial)
    gps = GPS(**gps_config)
    gps.initialize()
    logger.debug("Testing GPS data retrieval with valid response")
    data = gps.get_data()
    assert data is not None, "GPS data should not be None"
    assert abs(data["latitude"] - 12.576131) < 0.01, f"Latitude mismatch: {data['latitude']}"
    assert abs(data["longitude"] - 98.609053) < 0.01, f"Longitude mismatch: {data['longitude']}"
    assert data["speed"] == 10.0, f"Speed mismatch: {data['speed']}"
    logger.debug(f"GPS data retrieved: {data}")

def test_gps_get_data_no_fix(gps_config, mocker):
    mock_serial = Mock(write=Mock(), read=Mock(return_value=b"+CGPSINFO: ,,,,,,,,"))
    mocker.patch("serial.Serial", return_value=mock_serial)
    gps = GPS(**gps_config)
    gps.initialize()
    logger.debug("Testing GPS data retrieval with no fix")
    data = gps.get_data()
    assert data is None, "GPS data should be None for no fix"
    logger.debug("No GPS fix returned as expected")

def test_gps_close(gps_config, mocker):
    mock_serial = Mock(write=Mock(), read=Mock(return_value=b"OK"), close=Mock())
    mocker.patch("serial.Serial", return_value=mock_serial)
    gps = GPS(**gps_config)
    gps.initialize()
    logger.debug("Testing GPS close")
    gps.close()
    mock_serial.close.assert_called_once()
    assert gps.serial is None, "GPS serial should be None after close"
    logger.debug("GPS closed successfully")