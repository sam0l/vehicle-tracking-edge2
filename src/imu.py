import smbus
import time
import logging
from datetime import datetime
import numpy as np

class IMU:
    def __init__(self, i2c_bus, i2c_addresses, sample_rate, accel_range, gyro_range):
        self.logger = logging.getLogger(__name__)
        self.i2c_bus = i2c_bus
        self.i2c_addresses = [int(addr, 16) for addr in i2c_addresses]
        self.sample_rate = sample_rate
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        self.bus = None
        self.address = None
        self.last_position = None
        self.last_speed_ms = 0.0 # Speed in m/s for internal calculations
        self.current_heading_rad = 0.0 # Store heading in radians
        self.last_time = time.time()
        
        # MPU-6050 sensitivity scale factors (LSB per unit)
        self.accel_sens = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}
        self.gyro_sens = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}
        self.R_earth = 6371000.0 # Earth radius in meters

    def initialize(self):
        try:
            self.bus = smbus.SMBus(self.i2c_bus)
            for addr in self.i2c_addresses:
                try:
                    self.bus.write_byte_data(addr, 0x6B, 0x00)  # Wake up MPU-6050
                    self.address = addr
                    self.bus.write_byte_data(addr, 0x1C, {2: 0x00, 4: 0x08, 8: 0x10, 16: 0x18}[self.accel_range])
                    self.bus.write_byte_data(addr, 0x1B, {250: 0x00, 500: 0x08, 1000: 0x10, 2000: 0x18}[self.gyro_range])
                    self.logger.info(f"IMU initialized at address 0x{addr:02x}")
                    return True
                except Exception:
                    continue
            self.logger.error("Failed to find IMU at any address")
            return False
        except Exception as e:
            self.logger.error(f"IMU initialization error: {e}")
            return False

    def read_data(self):
        try:
            # Read raw accelerometer data (assuming MPU-6050 returns signed 16-bit)
            raw_ax = self._read_signed_word(0x3B)
            raw_ay = self._read_signed_word(0x3D)
            raw_az = self._read_signed_word(0x3F)
            
            # Read raw gyroscope data (assuming MPU-6050 returns signed 16-bit)
            raw_gx = self._read_signed_word(0x43)
            raw_gy = self._read_signed_word(0x45)
            raw_gz = self._read_signed_word(0x47)

            # Get sensitivity based on current range
            accel_s = self.accel_sens.get(self.accel_range, 16384.0)
            gyro_s = self.gyro_sens.get(self.gyro_range, 131.0)

            # Convert raw data to physical units (g for accel, deg/s for gyro)
            ax = raw_ax / accel_s
            ay = raw_ay / accel_s
            az = raw_az / accel_s
            gx = raw_gx / gyro_s
            gy = raw_gy / gyro_s
            gz = raw_gz / gyro_s
            
            accel_scaled = [ax, ay, az]
            gyro_scaled = [gx, gy, gz]

            current_time = time.time()
            dt = current_time - self.last_time
            if dt <= 0: # Avoid division by zero or negative dt if time changes
                dt = 1e-3 # nominal small dt
            
            # Simple speed calculation (m/s^2 * s = m/s)
            # Assuming az is forward/backward acceleration in g's, convert to m/s^2
            # This is a very strong assumption and likely needs calibration/orientation adjustment.
            forward_accel_ms2 = az * 9.80665 # Convert g to m/s^2 
            current_speed_ms = self.last_speed_ms + forward_accel_ms2 * dt
            current_speed_ms = max(0.0, current_speed_ms) # Speed should not be negative

            # Update heading using Z-axis gyro (yaw rate in deg/s)
            # Convert yaw rate to radians/s
            yaw_rate_rad_s = np.radians(gz)
            self.current_heading_rad += yaw_rate_rad_s * dt
            # Normalize heading to [-pi, pi]
            self.current_heading_rad = np.arctan2(np.sin(self.current_heading_rad), np.cos(self.current_heading_rad))

            # Basic stationary check (thresholds may need tuning)
            # Using scaled acceleration (in g's) and speed in m/s
            is_stationary = abs(az) < 0.05 and abs(current_speed_ms) < 0.1 # Example thresholds
            
            data = {
                "acceleration": accel_scaled, # In g's
                "gyro": gyro_scaled,         # In deg/s
                "speed": current_speed_ms,   # Speed in m/s from IMU integration
                "is_stationary": is_stationary,
                "timestamp": datetime.now().isoformat()
            }

            # Dead reckoning for position (highly experimental, prone to large drift)
            if self.last_position and dt > 0:
                displacement_m = current_speed_ms * dt
                
                lat1_rad = np.radians(self.last_position[0])
                lon1_rad = np.radians(self.last_position[1])
                
                # Simplified position calculation
                # For small distances, change in lat/lon can be approximated
                delta_lat_rad = (displacement_m * np.cos(self.current_heading_rad)) / self.R_earth
                delta_lon_rad = (displacement_m * np.sin(self.current_heading_rad)) / (self.R_earth * np.cos(lat1_rad))
                
                new_lat_rad = lat1_rad + delta_lat_rad
                new_lon_rad = lon1_rad + delta_lon_rad
                
                data["position"] = [
                    np.degrees(new_lat_rad),
                    np.degrees(new_lon_rad)
                ]
                # Update last_position with the new DR position for next iteration
                # Only if we exclusively rely on IMU for DR. If GPS updates frequently, 
                # self.last_position is updated by update_gps().
                # self.last_position = [np.degrees(new_lat_rad), np.degrees(new_lon_rad)] 

            self.last_speed_ms = current_speed_ms
            self.last_time = current_time
            return data
        except Exception as e:
            self.logger.error(f"IMU read error: {e}")
            return None

    def _read_signed_word(self, register):
        # Reads a 16-bit signed value from I2C (assuming MPU-6050, MSB first)
        # smbus.read_word_data often returns LSB first, so manual read might be safer or verify behavior.
        # This is a common way to read MPU6050 registers if smbus.read_word_data isn't directly suitable.
        try:
            high_byte = self.bus.read_byte_data(self.address, register)
            low_byte = self.bus.read_byte_data(self.address, register + 1)
            value = (high_byte << 8) | low_byte
            if value >= 0x8000: # If MSB is 1, it's a negative number in 2's complement
                return value - 0x10000
            return value
        except Exception as e:
            self.logger.debug(f"Failed to read word from register 0x{register:02x}: {e}")
            return 0 # Return 0 on error to avoid crashing, but this will affect data

    def update_gps(self, gps_data):
        if gps_data and gps_data.get('latitude') is not None and gps_data.get('longitude') is not None:
            self.last_position = [gps_data['latitude'], gps_data['longitude']]
            # When GPS updates, reset IMU heading if GPS provides course/bearing
            # or if we decide to re-align IMU DR with GPS fixes.
            # For now, just updating position. A more advanced fusion would use GPS speed/course.
            if 'speed' in gps_data and gps_data['speed'] is not None: # Assuming GPS speed is in m/s
                # Convert km/h from GPS (as per previous gps.py fix) to m/s for consistency with IMU speed
                # Note: gps.py was fixed to output km/h. Main.py might use this. This class uses m/s internally.
                # If gps_data['speed'] is already in m/s, this conversion is not needed.
                # For now, let's assume gps_data['speed'] is what VehicleTracker expects (km/h from gps.py).
                # This IMU class uses m/s. VehicleTracker needs to be aware of units.
                # self.last_speed_ms = (gps_data['speed'] * 1000) / 3600 # km/h to m/s
                pass # Let main.py handle speed unit consistency from different sources

            self.logger.debug(f"IMU updated with GPS position: {self.last_position}")

    def close(self):
        if self.bus:
            self.bus.write_byte_data(self.address, 0x6B, 0x01)  # Sleep MPU-6050
            self.bus.close()
            self.bus = None
            self.logger.info("IMU connection closed")