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
        self.last_speed = 0.0
        self.last_time = time.time()

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
            accel_data = [self.bus.read_word_data(self.address, reg) for reg in [0x3B, 0x3D, 0x3F]]
            accel = [(val / 16384.0) * self.accel_range for val in accel_data]
            gyro_data = [self.bus.read_word_data(self.address, reg) for reg in [0x43, 0x45, 0x47]]
            gyro = [(val / 131.0) * self.gyro_range for val in gyro_data]
            current_time = time.time()
            dt = current_time - self.last_time
            accel_z = accel[2]
            speed = self.last_speed + accel_z * dt
            speed = max(0.0, speed)
            is_stationary = abs(accel_z) < 0.1 and abs(speed) < 0.1
            data = {
                "acceleration": accel,
                "gyro": gyro,
                "speed": speed,
                "is_stationary": is_stationary,
                "timestamp": datetime.now().isoformat()
            }
            if self.last_position:
                displacement = speed * dt
                data["position"] = [
                    self.last_position[0] + displacement * np.cos(np.radians(gyro[2])),
                    self.last_position[1] + displacement * np.sin(np.radians(gyro[2]))
                ]
            self.last_speed = speed
            self.last_time = current_time
            return data
        except Exception as e:
            self.logger.error(f"IMU read error: {e}")
            return None

    def update_gps(self, gps_data):
        if gps_data and 'latitude' in gps_data and 'longitude' in gps_data:
            self.last_position = [gps_data['latitude'], gps_data['longitude']]
            self.logger.debug(f"IMU updated with GPS position: {self.last_position}")

    def close(self):
        if self.bus:
            self.bus.write_byte_data(self.address, 0x6B, 0x01)  # Sleep MPU-6050
            self.bus.close()
            self.bus = None
            self.logger.info("IMU connection closed")