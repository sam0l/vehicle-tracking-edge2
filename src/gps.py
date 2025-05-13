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
        self.last_known_speed_kmh = 0.0 # Store speed in a consistent unit

    def initialize(self):
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(self.power_delay)
            self.serial.write(b"AT+CGPS=1\r\n")  # Enable GPS
            time.sleep(self.agps_delay)  # Wait for AGPS
            response = self.serial.read(100).decode(errors='ignore')
            if "OK" not in response:
                self.logger.error(f"GPS initialization failed: {response}")
                # Attempt to ensure GPS is off if init failed partially
                try: self.serial.write(b"AT+CGPS=0\r\n")
                except: pass
                return False
            self.logger.info(f"GPS initialized on {self.port}")
            return True
        except Exception as e:
            self.logger.error(f"GPS initialization error: {e}")
            return False

    def get_data(self):
        try:
            self.serial.write(b"AT+CGNSSINFO\r\n")
            time.sleep(0.2) # Slightly increased for potentially longer responses
            response_bytes = self.serial.read(self.serial.in_waiting or 200) # Read available or up to 200
            response = response_bytes.decode(errors='replace') # Use replace for debugging

            # Example +CGNSSINFO: <lat>,<N/S>,<lon>,<E/W>,<date>,<time>,<alt>,<speed_kts>,<course>
            # This is a common 9-field format. Check modem documentation if issues arise.
            if "+CGNSSINFO:" in response:
                data_part = response.split("+CGNSSINFO:")[-1].strip()
                parts = data_part.split(",")
                
                # Expecting 9 parts for the format: lat,N/S,lon,E/W,date,time,alt,speed,course
                if len(parts) >= 9 and parts[0] and parts[2] and parts[6] and parts[7]: # Check essential fields
                    lat_dm = parts[0]
                    lat_deg = float(lat_dm[:2]) + (float(lat_dm[2:]) / 60.0)
                    if parts[1] == "S": lat_deg = -lat_deg

                    lon_dm = parts[2]
                    lon_deg = float(lon_dm[:3]) + (float(lon_dm[3:]) / 60.0)
                    if parts[3] == "W": lon_deg = -lon_deg
                    
                    # Assuming parts[6] is altitude in meters
                    alt_meters = float(parts[6]) if parts[6] else 0.0
                    
                    # Assuming parts[7] is speed in knots
                    speed_knots = float(parts[7]) if parts[7] else 0.0
                    speed_kmh = speed_knots * 1.852 # Convert knots to km/h
                    self.last_known_speed_kmh = speed_kmh

                    # Satellite count is not typically part of this specific 9-field CGNSSINFO response.
                    # If satellite info is needed, a different AT command (e.g., AT+CGNSSAT)
                    # or parsing NMEA sentences (like GGA, GSA) would be required.
                    # For now, we'll use the existing behavior which defaults to 0.
                    current_satellites = self.satellites # Retains existing behavior (default 0 or last value)
                    # To explicitly set to 0 if not found: current_satellites = 0
                    self.satellites = current_satellites

                    return {
                        "latitude": lat_deg,
                        "longitude": lon_deg,
                        "speed": speed_kmh, # Speed in km/h
                        "altitude": alt_meters,
                        "satellites": self.satellites,
                        "timestamp": datetime.now().isoformat()
                    }
                self.logger.debug(f"No valid GPS fix or insufficient parts in response: {data_part}")
                # Return last known speed if fix is lost, but other data is None
                return {"speed": self.last_known_speed_kmh, "latitude": None, "longitude": None}
            self.logger.warning(f"Invalid or no +CGNSSINFO in response: {response}")
            return {"speed": self.last_known_speed_kmh, "latitude": None, "longitude": None}
        except Exception as e:
            self.logger.error(f"GPS read error: {e}")
            return {"speed": self.last_known_speed_kmh, "latitude": None, "longitude": None} # Provide last speed on error

    def close(self):
        if self.serial:
            self.serial.write(b"AT+CGPS=0\r\n")  # Disable GPS
            self.serial.close()
            self.serial = None
            self.logger.info("GPS connection closed")