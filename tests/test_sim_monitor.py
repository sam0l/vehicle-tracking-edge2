import pytest
import logging
from src.sim_monitor import SimMonitor
from unittest.mock import Mock, patch
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture
def sim_config():
    return {
        "sim": {
            "port": "/dev/ttyUSB2",
            "baudrate": 115200,
            "check_interval": 60,
            "usage_file": "data_usage.json",
            "interfaces": ["ppp0"],
            "apn": "internet",
            "ussd_balance_code": "*221#",
            "modem_init_commands": ["AT+CFUN=1", "AT+CGATT=1", "AT+CREG=1"],
            "initialization_retries": 3
        },
        "backend": {
            "database_url": "postgresql://user:pass@localhost/db"
        }
    }

def test_sim_monitor_initialize_success(sim_config, mocker):
    mock_serial = Mock(
        write=Mock(),
        read=Mock(return_value=b"OK"),
        in_waiting=True
    )
    mocker.patch("serial.Serial", return_value=mock_serial)
    monitor = SimMonitor(sim_config)
    logger.debug(f"Testing SimMonitor initialization with port {sim_config['sim']['port']}")
    assert monitor.initialize(), "SimMonitor initialization failed"
    mock_serial.write.assert_any_call(b"AT+CPIN?\r\n")
    logger.debug("SimMonitor initialized successfully")

def test_sim_monitor_initialize_failure(sim_config, mocker):
    mocker.patch("serial.Serial", side_effect=Exception("Serial error"))
    monitor = SimMonitor(sim_config)
    logger.debug("Testing SimMonitor initialization failure")
    assert not monitor.initialize(), "SimMonitor initialization should have failed"
    logger.debug("SimMonitor initialization failed as expected")

def test_sim_monitor_check_balance(sim_config, mocker):
    mock_serial = Mock(
        write=Mock(),
        read=Mock(return_value=b'+CUSD: 1,"Balance: 10 USD",15'),
        in_waiting=True
    )
    mocker.patch("serial.Serial", return_value=mock_serial)
    monitor = SimMonitor(sim_config)
    monitor.initialize()
    logger.debug("Testing SimMonitor balance check")
    balance = monitor.check_sim_balance()
    assert balance is not None, "Balance should not be None"
    assert balance["balance"] == "Balance: 10 USD", f"Balance mismatch: {balance}"
    logger.debug(f"Balance retrieved: {balance}")

def test_sim_monitor_get_usage_stats(sim_config, mocker):
    monitor = SimMonitor(sim_config)
    monitor.usage_log.append({
        "timestamp": datetime.now().isoformat(),
        "bytes_sent": 1000,
        "bytes_received": 2000
    })
    logger.debug("Testing SimMonitor usage stats")
    stats = monitor.get_usage_stats("1d")
    assert stats["bytes_sent"] == 1000, f"Bytes sent mismatch: {stats['bytes_sent']}"
    assert stats["bytes_received"] == 2000, f"Bytes received mismatch: {stats['bytes_received']}"
    logger.debug(f"Usage stats: {stats}")

def test_sim_monitor_close(sim_config, mocker):
    mock_serial = Mock(
        write=Mock(),
        read=Mock(return_value=b"OK"),
        in_waiting=True,
        close=Mock()
    )
    mocker.patch("serial.Serial", return_value=mock_serial)
    monitor = SimMonitor(sim_config)
    monitor.initialize()
    logger.debug("Testing SimMonitor close")
    monitor.close()
    mock_serial.close.assert_called_once()
    assert monitor.serial is None, "Serial should be None after close"
    logger.debug("SimMonitor closed successfully")