import serial
import logging
import time
import requests
from datetime import datetime
import json
from collections import deque
import psutil
import subprocess
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.handlers.RotatingFileHandler('vehicle_tracker.log', maxBytes=10*1024*1024, backupCount=5)]
)
logger = logging.getLogger(__name__)

Base = declarative_base()

class UsageLog(Base):
    __tablename__ = 'usage_logs'
    id = sa.Column(sa.Integer, primary_key=True)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    bytes_sent = sa.Column(sa.BigInteger, nullable=False)
    bytes_received = sa.Column(sa.BigInteger, nullable=False)

class SimMonitor:
    def __init__(self, config):
        self.config = config['sim']
        self.backend_config = config['backend']
        self.port = self.config.get('port', "/dev/ttyUSB2")
        self.baudrate = self.config.get('baudrate', 115200)
        self.check_interval = self.config.get('check_interval', 60)
        self.usage_file = self.config.get('usage_file', "data_usage.json")
        self.interfaces = self.config.get('interfaces', ["ppp0"])
        self.apn = self.config.get('apn', "internet")
        self.ussd_balance_code = self.config.get('ussd_balance_code', "*221#")
        self.serial = None
        self.last_counters = self.get_current_counters()
        self.usage_log = deque(maxlen=10000)
        self.load_usage()
        # Database setup
        self.engine = sa.create_engine(self.backend_config['database_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def initialize(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            logger.info(f"Opened serial port {self.port}")
            for cmd in self.config.get('modem_init_commands', []):
                response = self.send_at_command(cmd)
                if not response or "ERROR" in response:
                    logger.error(f"Failed to execute {cmd}: {response}")
                    return False
            response = self.send_at_command("AT")
            if not response or "OK" not in response:
                logger.error(f"Modem not responding: {response}")
                return False
            sim_status = self.send_at_command("AT+CPIN?")
            if not sim_status or "READY" not in sim_status:
                logger.error(f"SIM card not ready: {sim_status}")
                return False
            for _ in range(self.config.get('initialization_retries', 3)):
                apn_cmd = f'AT+CGDCONT=1,"IP","{self.apn}"'
                if self.send_at_command(apn_cmd) and "ERROR" not in apn_cmd:
                    break
                time.sleep(1)
            logger.info("SIM Monitor initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SIM monitor: {e}")
            return False

    def send_at_command(self, command, timeout=2):
        if not self.serial:
            logger.error("Serial port not initialized")
            return None
        timeout = 10 if command.startswith("AT+CUSD") else timeout
        try:
            self.serial.write((command + "\r\n").encode())
            time.sleep(0.1)
            response = ""
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.serial.in_waiting:
                    response += self.serial.read(self.serial.in_waiting).decode(errors='ignore')
                if "OK" in response or "ERROR" in response:
                    break
                time.sleep(0.1)
            logger.debug(f"AT command '{command}' response: {response.strip()}")
            return response.strip()
        except Exception as e:
            logger.error(f"Error sending AT command '{command}': {e}")
            return None

    def load_usage(self):
        try:
            with open(self.usage_file, 'r') as f:
                self.usage_log = deque(json.load(f), maxlen=10000)
        except Exception:
            self.usage_log = deque(maxlen=10000)

    def save_usage(self):
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(list(self.usage_log), f)
            with self.Session() as session:
                for entry in self.usage_log:
                    log = UsageLog(
                        timestamp=datetime.fromisoformat(entry['timestamp']),
                        bytes_sent=entry['bytes_sent'],
                        bytes_received=entry['bytes_received']
                    )
                    session.add(log)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to save usage log: {e}")

    def log_data_usage(self, bytes_sent, bytes_received):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "bytes_sent": bytes_sent,
            "bytes_received": bytes_received
        }
        self.usage_log.append(entry)
        self.save_usage()

    def get_usage_stats(self, period="1d"):
        now = datetime.now()
        cutoff = now.timestamp() - {'1d': 86400, '1w': 7*86400, '1m': 30*86400}.get(period, 0)
        sent = received = 0
        points = []
        for entry in self.usage_log:
            ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
            if ts >= cutoff:
                sent += entry["bytes_sent"]
                received += entry["bytes_received"]
                points.append({"timestamp": entry["timestamp"], "bytes_sent": entry["bytes_sent"], "bytes_received": entry["bytes_received"]})
        return {"bytes_sent": sent, "bytes_received": received, "points": points}

    def get_current_counters(self):
        counters = psutil.net_io_counters(pernic=True)
        total_sent = total_recv = 0
        for iface in self.interfaces:
            if iface in counters:
                total_sent += counters[iface].bytes_sent
                total_recv += counters[iface].bytes_recv
        return {"bytes_sent": total_sent, "bytes_received": total_recv}

    def update_data_usage(self):
        current = self.get_current_counters()
        delta_sent = current["bytes_sent"] - self.last_counters["bytes_sent"]
        delta_recv = current["bytes_received"] - self.last_counters["bytes_received"]
        if delta_sent > 0 or delta_recv > 0:
            self.log_data_usage(delta_sent, delta_recv)
        self.last_counters = current

    def check_sim_balance(self):
        response = self.send_at_command(f'AT+CUSD=1,"{self.ussd_balance_code}"', timeout=10)
        if response and "+CUSD:" in response:
            try:
                parts = response.split('"')
                if len(parts) > 1:
                    return {"balance": parts[1]}
            except Exception as e:
                logger.error(f"Error parsing balance response: {e}")
        return None

    def get_network_info(self):
        if not self.serial:
            return None
        network_info = {}
        for cmd, key in [("AT+CREG?", "registration"), ("AT+COPS?", "operator"), ("AT+CGACT?", "connection")]:
            response = self.send_at_command(cmd)
            if response and f"+{cmd.split('?')[0].split('+')[1]}:" in response:
                network_info[key] = response
        return network_info if network_info else None

    def get_signal_strength(self):
        if not self.serial:
            return None
        signal = self.send_at_command("AT+CSQ")
        if signal and "+CSQ:" in signal:
            try:
                values = signal.split(":")[1].strip().split(",")
                signal_value = int(values[0])
                if signal_value < 99:
                    percentage = min(100, int(signal_value * 100 / 31))
                    return {"signal": signal_value, "percentage": percentage}
            except Exception as e:
                logger.error(f"Error parsing signal strength: {e}")
        return None

    def check_connectivity(self, host="8.8.8.8", port=53, timeout=3):
        import socket
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

    def reset_modem(self):
        self.send_at_command("AT+CFUN=0")
        time.sleep(2)
        self.send_at_command("AT+CFUN=1")

    def close(self):
        if self.serial:
            self.serial.close()
            self.serial = None
            logger.info("Serial connection closed")

def sim_process(config, data_queue):
    monitor = SimMonitor(config)
    if not monitor.initialize():
        logger.error("Failed to initialize SIM monitor")
        return
    try:
        while True:
            if not monitor.check_connectivity():
                logger.warning("PPP connection lost, reconnecting...")
                subprocess.run([monitor.config['ppp_setup']['connection_script']])
                time.sleep(10)
            balance_info = monitor.check_sim_balance()
            data_usage = monitor.get_usage_stats()
            network_info = monitor.get_network_info()
            signal_strength = monitor.get_signal_strength()
            if any([balance_info, data_usage, network_info, signal_strength]):
                data_queue.put(('sim', {
                    "balance": balance_info,
                    "data_usage": data_usage,
                    "network_info": network_info,
                    "signal_strength": signal_strength,
                    "timestamp": datetime.now().isoformat()
                }))
            time.sleep(monitor.check_interval)
    except KeyboardInterrupt:
        logger.info("SIM monitor stopped")
    finally:
        monitor.close()
