import yaml
import logging
import time
import json
import requests
import os
import cv2
import base64
import socket
import aiohttp
import asyncio
from datetime import datetime
from flask import Flask, jsonify
from multiprocessing import Process, Queue
import threading
import queue
import psutil
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker
from src.gps import GPS
from src.imu import IMU
from src.camera import Camera
from src.sign_detection import SignDetector
from src.sim_monitor import sim_process

app = Flask(__name__)

Base = declarative_base()

class Telemetry(Base):
    __tablename__ = 'telemetry'
    id = sa.Column(sa.Integer, primary_key=True)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    latitude = sa.Column(sa.Float)
    longitude = sa.Column(sa.Float)
    speed = sa.Column(sa.Float)
    satellites = sa.Column(sa.Integer)
    altitude = sa.Column(sa.Float)
    dead_reckoning = sa.Column(sa.Boolean)
    connection_status = sa.Column(sa.Boolean)

class Detection(Base):
    __tablename__ = 'detections'
    id = sa.Column(sa.Integer, primary_key=True)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    latitude = sa.Column(sa.Float)
    longitude = sa.Column(sa.Float)
    speed = sa.Column(sa.Float)
    sign_type = sa.Column(sa.String)
    confidence = sa.Column(sa.Float)
    image = sa.Column(sa.Text)
    connection_status = sa.Column(sa.Boolean)

class VehicleTracker:
    def __init__(self, config_path):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.setup_logging()
        self.logger.info(f"Loaded config: {json.dumps(self.config, indent=2)}")
        self.gps = GPS(**self.config['gps'])
        self.imu = IMU(**self.config['imu'])
        self.camera = Camera(**self.config['camera'])
        try:
            self.sign_detector = SignDetector(config_path)
        except Exception as e:
            self.logger.error(f"Failed to initialize SignDetector: {e}")
            self.sign_detector = None
        self.offline_data = []
        self.offline_file = self.config['logging']['offline_file']
        os.makedirs(os.path.dirname(self.offline_file) or '.', exist_ok=True)
        if not os.path.exists(self.offline_file):
            with open(self.offline_file, 'w') as f:
                json.dump([], f)
        self.camera_initialized = False
        self.app = app
        self.sim_lock = threading.Lock()
        self.setup_routes()
        self.last_telemetry_send_time = 0
        self.telemetry_interval = 1.0
        self.current_speed = 0.0
        self.speed_alpha = 0.8
        self.use_imu_speed = False
        self.last_speed_update = time.time()
        self.max_speed_age = 30
        self.recent_detections = {}
        self.detection_timeout = self.config['detection']['deduplication_timeout']
        self.detection_distance_threshold = self.config['detection']['distance_threshold']
        # Database setup
        self.engine = sa.create_engine(self.config['backend']['database_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.data_queue = Queue()
        self.batch = []
        self.last_batch_time = time.time()

    def setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.handlers.RotatingFileHandler(self.config['logging']['file'], maxBytes=10*1024*1024, backupCount=5),
                logging.StreamHandler()
            ]
        )

    def initialize(self, max_retries=5, retry_delay=5):
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            self.logger.info(f"Initialization attempt {attempt}/{max_retries}")
            import glob
            devices = glob.glob("/dev/video*")
            for device in devices:
                self.camera = Camera(device, self.config['camera']['width'], self.config['camera']['height'], self.config['camera']['fps'])
                if self.camera.initialize():
                    self.camera_initialized = True
                    break
            results = {
                'gps': self.gps.initialize(),
                'imu': self.imu.initialize(),
                'camera': self.camera_initialized
            }
            if results['gps'] and results['imu']:
                self.logger.info(f"Core components initialized (IMU address: 0x{self.imu.address:02x}, Camera: {self.camera_initialized})")
                return True
            self.logger.error(f"Initialization failed: {results}")
            if attempt < max_retries:
                time.sleep(retry_delay)
        return False

    def calculate_speed(self, gps_data, imu_data):
        current_time = time.time()
        satellites = gps_data.get('satellites', 0) if gps_data else 0
        self.speed_alpha = 0.8 if satellites >= 4 else 0.5
        if gps_data and 'speed' in gps_data and gps_data['speed'] is not None:
            gps_speed = gps_data['speed']
            self.last_speed_update = current_time
            if imu_data and 'speed' in imu_data:
                imu_speed = imu_data['speed']
                if imu_speed == 0 and imu_data.get('is_stationary', False):
                    self.current_speed = 0.0
                else:
                    self.current_speed = (self.speed_alpha * gps_speed) + ((1 - self.speed_alpha) * imu_speed)
                self.use_imu_speed = True
                self.imu.update_gps(gps_data)
            else:
                self.current_speed = gps_speed
            return round(self.current_speed, 2)
        elif imu_data and 'speed' in imu_data:
            self.last_speed_update = current_time
            self.current_speed = imu_data['speed']
            if imu_data.get('is_stationary', False):
                self.current_speed = 0.0
            return round(self.current_speed, 2)
        return round(self.current_speed, 2)

    def adjust_intervals(self):
        speed = self.current_speed
        cpu_load = psutil.cpu_percent()
        if speed < 5:
            self.config['logging']['interval']['camera'] = 1
            self.config['logging']['interval']['gps'] = 2
        elif cpu_load > 80:
            self.config['logging']['interval']['camera'] = 2
        else:
            self.config['logging']['interval']['camera'] = 0.5
            self.config['logging']['interval']['gps'] = 1

    def filter_duplicate_detections(self, signs, position, current_time):
        if not signs:
            return []
        if not position or position[0] == 0 or position[1] == 0:
            position = None
        speed = self.current_speed
        dynamic_threshold = max(0.0001, min(0.001, speed * 0.00001))
        filtered_signs = []
        for sign in signs:
            sign_type = sign['label']
            detection_key = sign_type
            should_include = True
            if detection_key in self.recent_detections:
                last_detection = self.recent_detections[detection_key]
                time_diff = current_time - last_detection['time']
                if time_diff < self.detection_timeout:
                    if position and last_detection['position']:
                        lat_diff = abs(position[0] - last_detection['position'][0])
                        lon_diff = abs(position[1] - last_detection['position'][1])
                        if lat_diff < dynamic_threshold and lon_diff < dynamic_threshold:
                            should_include = False
                            self.logger.debug(f"Filtering duplicate {sign_type}: Time {time_diff:.1f}s, Pos diff {lat_diff:.6f},{lon_diff:.6f}")
                    else:
                        should_include = False
            if should_include:
                filtered_signs.append(sign)
                self.recent_detections[detection_key] = {'time': current_time, 'position': position}
        for key in list(self.recent_detections.keys()):
            if current_time - self.recent_detections[key]['time'] > self.detection_timeout:
                del self.recent_detections[key]
        return filtered_signs

    async def send_async(self, payload, endpoint):
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.config['backend']['url']}{self.config['backend']['endpoint_prefix']}{endpoint}", json=payload, timeout=10) as response:
                if response.status == 200:
                    return True
                return False

    def send_data(self, data, frame=None):
        if not self.check_connectivity():
            self.log_offline(data)
            return False
        current_time = time.time()
        timestamp = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S").isoformat()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            data['image'] = base64.b64encode(buffer).decode('utf-8')
        self.batch.append(data)
        if len(self.batch) >= 5 or current_time - self.last_batch_time > 1:
            payload = {'batch': self.batch}
            loop = asyncio.get_event_loop()
            telemetry_sent = detection_sent = False
            if any(d.get('gps', {}).get('latitude') for d in self.batch):
                telemetry_sent = loop.run_until_complete(self.send_async(payload, self.config['backend']['telemetry_endpoint']))
            if any(d.get('signs') for d in self.batch):
                detection_sent = loop.run_until_complete(self.send_async(payload, self.config['backend']['detection_endpoint']))
            if telemetry_sent or detection_sent:
                with self.Session() as session:
                    for d in self.batch:
                        if 'gps' in d:
                            session.add(Telemetry(
                                timestamp=datetime.fromisoformat(timestamp),
                                latitude=d['gps'].get('latitude'),
                                longitude=d['gps'].get('longitude'),
                                speed=d['gps'].get('speed'),
                                satellites=d['gps'].get('satellites'),
                                altitude=d['gps'].get('altitude'),
                                dead_reckoning=d['gps'].get('dead_reckoning', False),
                                connection_status=self.check_connectivity()
                            ))
                        if 'signs' in d:
                            for sign in d['signs']:
                                session.add(Detection(
                                    timestamp=datetime.fromisoformat(timestamp),
                                    latitude=d['gps'].get('latitude', 0.0),
                                    longitude=d['gps'].get('longitude', 0.0),
                                    speed=d['gps'].get('speed', 0.0),
                                    sign_type=sign['label'],
                                    confidence=sign['confidence'],
                                    image=d.get('image'),
                                    connection_status=self.check_connectivity()
                                ))
                    session.commit()
                self.batch = []
                self.last_batch_time = current_time
                return True
            self.log_offline(data)
            return False
        return True

    def log_offline(self, data):
        self.offline_data.append(data)
        try:
            with open(self.offline_file, 'w') as f:
                json.dump(self.offline_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error logging offline data: {e}")

    def send_offline_data(self):
        if not self.offline_data:
            return
        for data in self.offline_data[:]:
            if self.send_data(data):
                self.offline_data.remove(data)
        try:
            with open(self.offline_file, 'w') as f:
                json.dump(self.offline_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error updating offline data file: {e}")

    def check_connectivity(self, host="8.8.8.8", port=53, timeout=3):
        try:
            socket.setdefaulttimeout(timeout)
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
            return True
        except socket.error:
            return False

    def setup_routes(self):
        @self.app.route('/api/data-usage')
        def get_data_usage():
            with self.sim_lock:
                return jsonify({
                    '1d': self.sim_monitor.get_usage_stats('1d'),
                    '1w': self.sim_monitor.get_usage_stats('1w'),
                    '1m': self.sim_monitor.get_usage_stats('1m')
                })

    def gps_process(self):
        last_gps = 0
        while True:
            current_time = time.time()
            if current_time - last_gps >= self.config['logging']['interval']['gps']:
                try:
                    gps_data = self.gps.get_data()
                    if gps_data:
                        self.data_queue.put(('gps', gps_data))
                except Exception as e:
                    self.logger.error(f"GPS error: {e}")
                last_gps = current_time
            time.sleep(0.01)

    def imu_process(self):
        last_imu = 0
        while True:
            current_time = time.time()
            if current_time - last_imu >= self.config['logging']['interval']['imu']:
                try:
                    imu_data = self.imu.read_data()
                    if imu_data:
                        self.data_queue.put(('imu', imu_data))
                except Exception as e:
                    self.logger.error(f"IMU error: {e}")
                last_imu = current_time
            time.sleep(0.001)

    def camera_process(self):
        last_camera = last_camera_init = 0
        camera_init_interval = 30
        frame_queue = Queue(maxsize=10)
        def capture_thread():
            while self.camera_initialized:
                frame = self.camera.get_frame()
                if frame is not None:
                    frame_queue.put(frame)
                time.sleep(0.01)
        capture_t = threading.Thread(target=capture_thread, daemon=True)
        capture_t.start()
        while True:
            current_time = time.time()
            if not self.camera_initialized and current_time - last_camera_init >= camera_init_interval:
                self.camera_initialized = self.camera.initialize()
                last_camera_init = current_time
            if self.camera_initialized and current_time - last_camera >= self.config['logging']['interval']['camera']:
                try:
                    frame = frame_queue.get(timeout=0.2)
                    if self.sign_detector:
                        signs = self.sign_detector.detect(frame)
                        if signs:
                            self.data_queue.put(('signs', signs, frame))
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Camera error: {e}")
                last_camera = current_time
            time.sleep(0.01)

    def data_transmission_process(self):
        last_known_position = None
        using_dead_reckoning = False
        dead_reckoning_start_time = 0
        max_dead_reckoning_time = 300
        consecutive_gps_failures = 0
        max_gps_failures = 20
        consecutive_imu_failures = 0
        max_imu_failures = 5
        while True:
            try:
                data_type, *data = self.data_queue.get(timeout=1)
                current_time = time.time()
                self.adjust_intervals()
                data_packet = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                gps_failure = imu_failure = False
                if data_type == 'gps':
                    gps_data = data[0]
                    data_packet['gps'] = gps_data
                    last_known_position = (gps_data['latitude'], gps_data['longitude'])
                    consecutive_gps_failures = 0
                    using_dead_reckoning = False
                elif data_type == 'imu':
                    imu_data = data[0]
                    data_packet['imu'] = imu_data
                    consecutive_imu_failures = 0
                    if 'gps' in data_packet:
                        self.imu.update_gps(data_packet['gps'])
                    if consecutive_gps_failures > 0 and imu_data.get('position'):
                        using_dead_reckoning = True
                        dead_reckoning_start_time = current_time
                elif data_type == 'signs':
                    signs, frame = data
                    position = (data_packet.get('gps', {}).get('latitude'), data_packet.get('gps', {}).get('longitude'))
                    filtered_signs = self.filter_duplicate_detections(signs, position, current_time)
                    if filtered_signs:
                        data_packet['signs'] = filtered_signs
                        self.send_data(data_packet, frame)
                elif data_type == 'sim':
                    sim_data = data[0]
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.send_async(sim_data, self.config['backend']['sim_data_endpoint']))
                if 'gps' in data_packet or 'imu' in data_packet:
                    speed = self.calculate_speed(data_packet.get('gps'), data_packet.get('imu'))
                    if 'gps' in data_packet:
                        data_packet['gps']['speed'] = speed
                    elif 'imu' in data_packet:
                        data_packet['imu']['speed'] = speed
                    if gps_failure and not imu_failure and using_dead_reckoning:
                        if current_time - dead_reckoning_start_time > max_dead_reckoning_time:
                            using_dead_reckoning = False
                            self.gps.initialize()
                        elif data_packet.get('imu', {}).get('position'):
                            data_packet['gps'] = {
                                'latitude': data_packet['imu']['position'][0],
                                'longitude': data_packet['imu']['position'][1],
                                'speed': data_packet['imu']['speed'],
                                'dead_reckoning': True
                            }
                    self.send_data(data_packet)
                self.send_offline_data()
                if consecutive_gps_failures >= max_gps_failures:
                    self.gps.initialize()
                    consecutive_gps_failures = 0
                if consecutive_imu_failures >= max_imu_failures:
                    self.imu.initialize()
                    consecutive_imu_failures = 0
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Transmission error: {e}")

    def monitor_processes(self, processes):
        while True:
            for p in processes:
                if not p.is_alive():
                    self.logger.error(f"Process {p.name} died, restarting...")
                    p = Process(target=p._target, args=p._args)
                    p.start()
            time.sleep(10)

    def run(self):
        if not self.initialize():
            self.logger.error("Initialization failed, exiting")
            return
        processes = [
            Process(target=self.gps_process),
            Process(target=self.imu_process),
            Process(target=self.camera_process),
            Process(target=self.data_transmission_process),
            Process(target=sim_process, args=(self.config, self.data_queue))
        ]
        for p in processes:
            p.daemon = True
            p.start()
        flask_thread = threading.Thread(target=self.app.run, kwargs={'host': '0.0.0.0', 'port': self.config['api']['port']})
        flask_thread.daemon = True
        flask_thread.start()
        monitor_thread = threading.Thread(target=self.monitor_processes, args=(processes,))
        monitor_thread.daemon = True
        monitor_thread.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
        finally:
            for p in processes:
                p.terminate()
            self.cleanup()

    def cleanup(self):
        try:
            self.gps.close()
            self.imu.close()
            self.camera.close()
            if self.sign_detector:
                self.sign_detector.close()
            self.logger.info("Resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    tracker = VehicleTracker("config/config.yaml")
    tracker.run()