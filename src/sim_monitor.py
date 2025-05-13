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
import socket

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
        self.config_main = config # Keep main config for backend db_url
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
        self.usage_log = deque(maxlen=10000) # In-memory log for recent raw entries
        self.last_db_log_timestamp = None # Track last timestamp logged to DB
        self.load_usage_from_file() # Load historical from JSON file
        # Database setup
        db_url = self.config_main['backend']['database_url']
        self.engine = sa.create_engine(db_url)
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
            logger.error("Serial port not initialized for AT command")
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

    def load_usage_from_file(self):
        try:
            with open(self.usage_file, 'r') as f:
                # Loads a list of dicts, each is a delta measurement
                loaded_entries = json.load(f)
                self.usage_log = deque(loaded_entries, maxlen=10000)
                if self.usage_log:
                    # Try to set last_db_log_timestamp to the latest from file to avoid re-logging old file data
                    try:
                        self.last_db_log_timestamp = datetime.fromisoformat(self.usage_log[-1]['timestamp'])
                    except: pass # Ignore if format is bad or entry doesn't exist
            logger.info(f"Loaded {len(self.usage_log)} usage entries from {self.usage_file}")
        except FileNotFoundError:
            logger.info(f"Usage file {self.usage_file} not found, starting fresh.")
            self.usage_log = deque(maxlen=10000)
        except Exception as e:
            logger.error(f"Error loading usage from {self.usage_file}: {e}")
            self.usage_log = deque(maxlen=10000)

    def save_usage_to_file_and_db(self, new_log_entry=None):
        # new_log_entry is the single, new delta to be added and potentially saved to DB
        # The full self.usage_log (in-memory deque) is always saved to JSON file for persistence.

        # Save the full deque to JSON file
        try:
            with open(self.usage_file, 'w') as f:
                json.dump(list(self.usage_log), f, indent=2) # Save with indent for readability
        except Exception as e:
            logger.error(f"Failed to save usage log to file {self.usage_file}: {e}")

        # Save only the new_log_entry to database if it's provided
        if new_log_entry:
            try:
                entry_ts = datetime.fromisoformat(new_log_entry['timestamp'])
                # Optional: Check if this timestamp is newer than last saved to avoid duplicates from restarts
                if self.last_db_log_timestamp and entry_ts <= self.last_db_log_timestamp:
                    logger.debug(f"Skipping DB log for entry already logged or older: {entry_ts}")
                    return

                with self.Session() as session:
                    db_log = UsageLog(
                        timestamp=entry_ts,
                        bytes_sent=new_log_entry['bytes_sent'],
                        bytes_received=new_log_entry['bytes_received']
                    )
                    session.add(db_log)
                    session.commit()
                    self.last_db_log_timestamp = entry_ts # Update last saved timestamp
                    logger.debug(f"Logged new data usage to DB: {new_log_entry}")
            except Exception as e:
                logger.error(f"Failed to save new usage entry to database: {e}")

    def log_new_data_usage_delta(self, delta_sent, delta_received):
        # This logs a new DELTA of usage.
        entry = {
            "timestamp": datetime.now().isoformat(),
            "bytes_sent": delta_sent,
            "bytes_received": delta_received
        }
        self.usage_log.append(entry) # Add to in-memory deque
        self.save_usage_to_file_and_db(new_log_entry=entry) # Save deque to file, and this new entry to DB

    def get_usage_stats_for_period(self, period_key="1d"):
        # Calculates totals from the self.usage_log (deque of deltas)
        now = datetime.now()
        cutoff = now.timestamp() - {'1d': 86400, '1w': 7*86400, '1m': 30*86400}.get(period_key, 0)
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

    def update_data_usage_counters(self):
        # This should be called periodically to check psutil counters and log deltas.
        current = self.get_current_counters()
        if self.last_counters is None: # First call after init or if psutil failed previously
            self.last_counters = current
            logger.info("Initialized psutil data counters.")
            return

        # Ensure counters are not smaller (e.g. after interface reset or psutil bug)
        delta_sent = current["bytes_sent"] - self.last_counters["bytes_sent"]
        delta_recv = current["bytes_received"] - self.last_counters["bytes_received"]

        if delta_sent < 0: delta_sent = current["bytes_sent"] # Counter reset, take current as delta
        if delta_recv < 0: delta_recv = current["bytes_received"] # Counter reset

        if delta_sent > 0 or delta_recv > 0:
            self.log_new_data_usage_delta(delta_sent, delta_recv)
            logger.info(f"Logged data usage delta: Sent={delta_sent}, Recv={delta_recv}")
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

    def check_connectivity(self):
        # Uses network configuration from the main config object
        cfg_net = self.config_main.get('network', {})
        host = cfg_net.get('ping_test_host', "8.8.8.8")
        port = cfg_net.get('ping_test_port', 53)
        timeout = cfg_net.get('ping_timeout_s', 3)
        try:
            # Use socket.create_connection for a non-blocking connect with timeout
            # This avoids using socket.setdefaulttimeout(), which is global.
            conn = socket.create_connection((host, port), timeout=timeout)
            conn.close()
            return True
        except (socket.error, socket.timeout) as e: # Specify exceptions
            logger.debug(f"Connectivity check to {host}:{port} failed: {e}")
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
        # Initial update of data usage counters
        monitor.update_data_usage_counters()

        while True:
            # Periodically update data usage based on psutil counters
            monitor.update_data_usage_counters()

            if not monitor.check_connectivity():
                logger.warning("PPP connection lost, attempting to reconnect...")
                try:
                    connection_script = monitor.config.get('ppp_setup', {}).get('connection_script')
                    if connection_script:
                        subprocess.run([connection_script], check=True)
                        logger.info("PPP connection script executed.")
                        time.sleep(10) # Wait for connection to establish
                    else:
                        logger.warning("No PPP connection script defined in config.")
                except subprocess.CalledProcessError as e_sub:
                    logger.error(f"PPP connection script failed: {e_sub}")
                except FileNotFoundError:
                    logger.error(f"PPP connection script not found at: {connection_script}")
                except Exception as e_ppp:
                    logger.error(f"Error running PPP connection script: {e_ppp}")
            else:
                balance_info = monitor.check_sim_balance()
                usage_stats_current_interval = monitor.get_usage_stats_for_period('interval') # Special key for "since last check"
                # This would require get_usage_stats_for_period to handle 'interval' perhaps by only summing last N entries
                # For simplicity, let's send the standard '1d' usage or what's configured for reporting.
                # The data_queue payload should be what the backend expects for 'sim' data packets.
                # The example below sends cumulative stats which might be large.
                # Consider what actual data needs to be sent regularly.

                network_info = monitor.get_network_info()
                signal_strength = monitor.get_signal_strength()

                # Prepare data packet for main data_queue
                sim_data_packet = {
                    "timestamp": datetime.now().isoformat(),
                    "balance": balance_info,
                    # Send total usage for a default period, e.g., '1d' or make configurable
                    "data_usage_report": monitor.get_usage_stats_for_period(monitor.config.get('usage_report_period', '1d')),
                    "network_info": network_info,
                    "signal_strength": signal_strength,
                    "modem_port_status": "connected" if monitor.serial and monitor.serial.is_open else "disconnected"
                }

                if any(sim_data_packet.values()): # If any data was gathered
                    data_queue.put(('sim', sim_data_packet))

            time.sleep(monitor.check_interval) # Main operational loop sleep
    except KeyboardInterrupt:
        logger.info("SIM monitor stopped")
    finally:
        monitor.close()
