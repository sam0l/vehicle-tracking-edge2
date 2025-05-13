import serial
import time
import logging
from datetime import datetime

class GPS:
    def __init__(self, port, baudrate, timeout, power_delay, agps_delay):
        self.logger = logging.getLogger(__name__)
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.power_delay = power_delay
        self.agps_delay = agps_delay
        self.serial = None
        self.satellites = 0

    def initialize(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(self.power_delay)
            self.serial.write(b"AT+CGPS=1\r\n")  # Enable GPS
            time.sleep(self.agps_delay)  # Wait for AGPS
            response = self.serial.read(100).decode(errors='ignore')
            if "OK" not in response:
                self.logger.error(f"GPS initialization failed: {response}")
                return False
            self.logger.info(f"GPS initialized on {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"GPS initialization error: {e}")
            return False

    def get_data(self):
        try:
            self.serial.write(b"AT+CGPSINFO\r\n")
            time.sleep(0.1)
            response = self.serial.read(200).decode(errors='ignore')
            if "+CGPSINFO:" in response:
                parts = response.split(":")[1].strip().split(",")
                if len(parts) >= 8 and parts[0]:  # Valid fix
                    lat = float(parts[0][:2]) + float(parts[0][2:]) / 60
                    lat = lat if parts[1] == "N" else -lat
                    lon = float(parts[2][:3]) + float(parts[2][3:]) / 60
                    lon = lon if parts[3] == "E" else -lon
                    speed = float(parts[6]) if parts[6] else 0.0
                    self.satellites = int(self.serial.read(100).decode(errors='ignore').split(",")[2]) if "+CSQ:" in response else self.satellites
                    return {
                        "latitude": lat,
                        "longitude": lon,
                        "speed": speed,
                        "altitude": float(parts[5]) if parts[5] else 0.0,
                        "satellites": self.satellites,
                        "timestamp": datetime.now().isoformat()
                    }
                self.logger.debug("No valid GPS fix")
                return None
            self.logger.warning(f"Invalid GPS response: {response}")
            return None
        except Exception as e:
            self.logger.error(f"GPS read error: {e}")
            return None

    def close(self):
        if self.serial:
            self.serial.write(b"AT+CGPS=0\r\n")  # Disable GPS
            self.serial.close()
            self.serial = None
            self.logger.info("GPS connection closed")