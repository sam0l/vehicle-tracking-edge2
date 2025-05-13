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
import glob
from datetime import datetime
from flask import Flask, jsonify
from multiprocessing import Process, Queue, Pipe
import threading
from queue import Empty as QueueEmpty, Full as QueueFull # Standard library queue exceptions
import psutil
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, sessionmaker
from logging.handlers import RotatingFileHandler # Corrected import

from src.gps import GPS
from src.imu import IMU
from src.camera import Camera
from src.sign_detection import SignDetector
from src.sim_monitor import sim_process # Assuming sim_process is a function target

app = Flask(__name__)
Base = declarative_base()

# --- SQLAlchemy Models ---
class Telemetry(Base):
    __tablename__ = 'telemetry'
    id = sa.Column(sa.Integer, primary_key=True)
    timestamp = sa.Column(sa.DateTime, nullable=False)
    latitude = sa.Column(sa.Float)
    longitude = sa.Column(sa.Float)
    speed = sa.Column(sa.Float) # Assumed km/h
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
    speed = sa.Column(sa.Float) # Assumed km/h
    sign_type = sa.Column(sa.String)
    confidence = sa.Column(sa.Float)
    image = sa.Column(sa.Text) # Base64 encoded image
    connection_status = sa.Column(sa.Boolean)

class VehicleTracker:
    def __init__(self, config_path):
        self.config_path = config_path # Store for potential use in sub-processes
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_logging()
        self.logger.info(f"Loaded config: {json.dumps(self.config, indent=2)}")

        # Pipes for dynamic interval updates
        self.gps_interval_pipe_recv, self.gps_interval_pipe_send = Pipe(duplex=False)
        self.imu_interval_pipe_recv, self.imu_interval_pipe_send = Pipe(duplex=False)
        self.camera_interval_pipe_recv, self.camera_interval_pipe_send = Pipe(duplex=False)

        self.offline_data = []
        self.offline_file = self.config['logging']['offline_file']
        os.makedirs(os.path.dirname(self.offline_file) or '.', exist_ok=True)
        # Load existing offline data if any (JSON Lines format)
        if os.path.exists(self.offline_file):
            try:
                with open(self.offline_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                self.offline_data.append(json.loads(line))
                            except json.JSONDecodeError as e_json:
                                self.logger.error(f"Error decoding JSON line from offline file: {e_json} - Line: '{line}'")
            except Exception as e_file:
                self.logger.error(f"Error reading offline file {self.offline_file}: {e_file}")
        self.logger.info(f"Loaded {len(self.offline_data)} items from offline log.")

        self.app = app
        self.sim_lock = threading.Lock() # If needed for shared SimMonitor access by API

        # API related - self.sim_monitor_instance might be created if API needs direct calls
        # self.sim_monitor_instance = SimMonitor(self.config) # For P30 fix
        self.setup_routes()

        # Load app settings from config
        app_cfg = self.config.get('app_settings', {})
        self.current_speed_kmh = 0.0 
        self.last_speed_update_time = time.time()
        self.max_speed_age_s = app_cfg.get('speed_staleness_s', 30)
        self.speed_smoothing_alpha = app_cfg.get('speed_alpha', 0.8)
        self.speed_alpha_min_satellites = app_cfg.get('speed_alpha_min_satellites', 4)
        self.speed_alpha_low_sat_value = app_cfg.get('speed_alpha_low_sat_value', 0.5)
        self.process_monitor_interval_s = app_cfg.get('process_monitor_interval_s', 10)

        self.recent_detections = {}
        detection_cfg = self.config.get('detection', {})
        self.detection_timeout_s = detection_cfg.get('deduplication_timeout', 10)
        self.detection_dist_thresh_deg = detection_cfg.get('distance_threshold', 0.001)

        self.engine = sa.create_engine(self.config['backend']['database_url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self.data_queue = Queue() # Generic data queue from processes
        self.batch_outgoing_data = []
        self.last_batch_send_time = time.time()
        backend_cfg = self.config.get('backend', {})
        self.batch_size_limit = backend_cfg.get('batch_size', 5)
        self.batch_time_limit_s = backend_cfg.get('batch_time_s', 1.0)
        
        self.active_processes_details = [] # For monitoring
        self.last_sim_data_usage_report = {} # For P30: Cache last SIM data usage for API
        self.sim_data_lock = threading.Lock() # For P30: Lock for accessing cached SIM data

    def setup_logging(self):
        if not logging.getLogger().handlers: # Idempotency check
            log_level_str = self.config['logging']['level'].upper()
            log_level = getattr(logging, log_level_str, logging.INFO)
            logging.basicConfig(
                level=log_level,
                format=self.config['logging']['format'],
                handlers=[
                    RotatingFileHandler(self.config['logging']['file'], maxBytes=10*1024*1024, backupCount=5),
                    logging.StreamHandler()
                ]
            )
            self.logger.info(f"Logging configured by VehicleTracker at level {log_level_str}.")
        else:
            self.logger.info("Logging already configured.")

    def initialize_main_components(self):
        # Initialize components needed by the main thread for pre-checks or shared info
        # Sensor objects themselves are created within their respective processes.
        # This method is for validating if such components *could* be initialized.
        self.logger.info("Performing initial component checks (not starting sensors here)...")
        # Example: Check if a SignDetector could be initialized
        try:
            # This doesn't hold the instance, just checks creation
            _ = SignDetector(self.config_path) 
            self.logger.info("Initial SignDetector check: OK (can be created).")
        except Exception as e:
            self.logger.warning(f"Initial SignDetector check: Failed ({e}). Camera process will attempt its own.")
        
        # Example: Try to open and close a camera for initial validation
        cam_check_ok = False
        try:
            cam_conf = self.config['camera']
            test_cam = Camera(cam_conf.get('device_id', "/dev/video0"), cam_conf['width'], cam_conf['height'], cam_conf['fps'])
            if test_cam.initialize():
                self.logger.info("Initial camera hardware check: OK.")
                cam_check_ok = True
            else:
                self.logger.warning("Initial camera hardware check: Failed to open.")
            test_cam.close()
        except Exception as e:
            self.logger.warning(f"Initial camera hardware check: Error ({e}).")
        
        # Similar checks for GPS/IMU could be done if they don't auto-start hardware
        # For now, assume their process-local init is sufficient.
        return cam_check_ok # Return status of critical pre-checks

    def calculate_speed_kmh(self, gps_payload, imu_payload):
        # gps_payload['speed'] is km/h (from gps.py)
        # imu_payload['speed'] is m/s (from imu.py)
        current_time = time.time()
        
        gps_speed_kmh = gps_payload.get('speed') if gps_payload else None
        imu_speed_ms = imu_payload.get('speed') if imu_payload else None
        imu_is_stationary = imu_payload.get('is_stationary', False) if imu_payload else False

        imu_speed_kmh = (imu_speed_ms * 3.6) if imu_speed_ms is not None else None

        final_speed_kmh = self.current_speed_kmh # Default to last known

        stationary_cfg = self.config.get('stationary_detection', {})
        gps_stationary_speed_thresh_kmh = stationary_cfg.get('speed_threshold_kmh', 5)

        if gps_speed_kmh is not None and gps_payload.get('latitude') is not None: # Valid GPS fix
            self.last_speed_update_time = current_time
            # Use configured alpha based on satellite count
            alpha = self.speed_smoothing_alpha if gps_payload.get('satellites', 0) >= self.speed_alpha_min_satellites else self.speed_alpha_low_sat_value
            if imu_speed_kmh is not None:
                final_speed_kmh = (alpha * gps_speed_kmh) + ((1 - alpha) * imu_speed_kmh)
                if imu_is_stationary and gps_speed_kmh < gps_stationary_speed_thresh_kmh: final_speed_kmh = 0.0
            else:
                final_speed_kmh = gps_speed_kmh
        elif imu_speed_kmh is not None:
            self.last_speed_update_time = current_time
            final_speed_kmh = imu_speed_kmh
            if imu_is_stationary: final_speed_kmh = 0.0
        else: # No new data
            if current_time - self.last_speed_update_time > self.max_speed_age_s:
                final_speed_kmh = 0.0 # Stale, assume stopped

        self.current_speed_kmh = round(final_speed_kmh, 2)
        return self.current_speed_kmh

    def adjust_intervals(self):
        speed = self.current_speed_kmh
        cpu = psutil.cpu_percent()
        cfg_intervals = self.config['logging']['interval']
        low_speed_intervals_cfg = self.config.get('low_speed_intervals', {})
        high_cpu_intervals_cfg = self.config.get('high_cpu_intervals', {})

        new_cam_interval = cfg_intervals['camera']
        new_gps_interval = cfg_intervals['gps']

        if speed < 5:
            new_cam_interval = low_speed_intervals_cfg.get('camera', 1.0)
            new_gps_interval = low_speed_intervals_cfg.get('gps', 2.0)
        elif cpu > 80:
            if abs(new_cam_interval - cfg_intervals['camera']) < 0.01: # Only if not already changed by low speed
                 new_cam_interval = high_cpu_intervals_cfg.get('camera', 2.0)
        
        try: self.camera_interval_pipe_send.send(new_cam_interval)
        except Exception as e: self.logger.error(f"Pipe send CAM interval error: {e}")
        try: self.gps_interval_pipe_send.send(new_gps_interval)
        except Exception as e: self.logger.error(f"Pipe send GPS interval error: {e}")
        # IMU interval not dynamically changed in this logic

    def filter_duplicate_detections(self, signs, gps_data, current_time_sec):
        if not signs: return []
        pos = (gps_data['latitude'], gps_data['longitude']) if gps_data and gps_data.get('latitude') is not None else None
        
        filtered = []
        for sign in signs:
            key = sign['label']
            if key in self.recent_detections:
                last = self.recent_detections[key]
                time_diff = current_time_sec - last['time']
                if time_diff < self.detection_timeout_s:
                    if pos and last.get('pos'):
                        # Approx distance check (squared Euclidean on lat/lon)
                        d_sq = (pos[0] - last['pos'][0])**2 + (pos[1] - last['pos'][1])**2
                        if d_sq < (self.detection_dist_thresh_deg**2):
                            continue # Skip duplicate
                    else: # No position, rely on time
                        continue
            filtered.append(sign)
            self.recent_detections[key] = {'time': current_time_sec, 'pos': pos}
        
        # Cleanup old
        for k in list(self.recent_detections.keys()):
            if current_time_sec - self.recent_detections[k]['time'] > self.detection_timeout_s:
                del self.recent_detections[k]
        return filtered

    async def _send_to_backend_async(self, payload, endpoint_suffix):
        url = f"{self.config['backend']['url']}{self.config['backend']['endpoint_prefix']}{endpoint_suffix}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    return resp.status == 200
        except Exception as e:
            self.logger.error(f"Async send to {url} failed: {e}")
            return False

    def _get_or_create_event_loop(self):
        try:
            return asyncio.get_event_loop()
        except RuntimeError: # No current event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def _send_single_packet(self, packet_content, associated_frame=None):
        if not self.check_connectivity():
            self.log_offline_data(packet_content, associated_frame)
            return False

        if associated_frame is not None and 'image' not in packet_content:
            try:
                quality = self.config['camera'].get('jpeg_quality', 50)
                _, buf = cv2.imencode('.jpg', associated_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                packet_content['image'] = base64.b64encode(buf).decode('utf-8')
            except Exception as e: self.logger.error(f"JPEG encode error: {e}")
        
        loop = self._get_or_create_event_loop()
        sent_ok = False
        
        # Determine endpoint based on content
        if 'signs' in packet_content:
            if loop.run_until_complete(self._send_to_backend_async(packet_content, self.config['backend']['detection_endpoint'])):
                sent_ok = True
        # Telemetry might be part of detection packet or standalone
        elif 'gps' in packet_content or 'imu' in packet_content:
            if loop.run_until_complete(self._send_to_backend_async(packet_content, self.config['backend']['telemetry_endpoint'])):
                sent_ok = True
        
        if sent_ok: self._log_to_db(packet_content)
        else: self.log_offline_data(packet_content, associated_frame)
        return sent_ok

    def _log_to_db(self, packet):
        with self.Session() as session:
            try:
                ts = datetime.fromisoformat(packet["timestamp"].replace(" ", "T"))
                conn_status = self.check_connectivity()

                if 'gps' in packet:
                    gps = packet['gps']
                    session.add(Telemetry(timestamp=ts, latitude=gps.get('latitude'), longitude=gps.get('longitude'),
                                          speed=gps.get('speed'), satellites=gps.get('satellites'), altitude=gps.get('altitude'),
                                          dead_reckoning=gps.get('dead_reckoning', False), connection_status=conn_status))
                
                if 'signs' in packet:
                    gps_for_det = packet.get('gps', {})
                    for sign in packet['signs']:
                        session.add(Detection(timestamp=ts, latitude=gps_for_det.get('latitude'), longitude=gps_for_det.get('longitude'),
                                              speed=gps_for_det.get('speed', self.current_speed_kmh), sign_type=sign['label'],
                                              confidence=sign['confidence'], image=packet.get('image'), connection_status=conn_status))
                session.commit()
            except Exception as e:
                self.logger.error(f"DB log error: {e}"); session.rollback()

    def log_offline_data(self, data, frame=None):
        entry = data.copy()
        if frame is not None and 'image' not in entry: # If no image already, try to add a low-quality one
            try:
                quality = self.config['camera'].get('offline_jpeg_quality', 20)
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                entry['image_offline_b64'] = base64.b64encode(buf).decode('utf-8')
            except Exception as e: self.logger.warning(f"Offline encode error: {e}")
        
        self.offline_data.append(entry) # Add to in-memory list
        try:
            # Append to file (JSON Lines format)
            with open(self.offline_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.error(f"Offline log append error: {e}")

    def send_offline_data_batch(self):
        if not self.offline_data or not self.check_connectivity(): return
        
        sent_indices = []
        for i, entry in enumerate(list(self.offline_data)): # Iterate a copy for safe removal later
            packet = entry.copy()
            if 'image_offline_b64' in packet: # Promote offline image for sending
                packet['image'] = packet.pop('image_offline_b64')
            
            # Here, associated_frame is None because image data is already in packet if it exists
            if self._send_single_packet(packet, associated_frame=None):
                sent_indices.append(i)
            else: break # Stop if one fails, to retry later
        
        if sent_indices: # If anything was sent and needs to be removed from log
            for i in sorted(sent_indices, reverse=True):
                del self.offline_data[i]
            
            # Rewrite the offline file with the remaining data
            try:
                with open(self.offline_file, 'w') as f:
                    for item in self.offline_data:
                        f.write(json.dumps(item) + '\n')
                self.logger.info(f"Sent {len(sent_indices)} offline items. {len(self.offline_data)} remaining.")
            except Exception as e:
                self.logger.error(f"Offline log rewrite error after sending: {e}")

    def check_connectivity(self, host=None, port=None, timeout_s=None):
        cfg_net = self.config['network']
        host = host or cfg_net.get('ping_test_host', "8.8.8.8")
        port = port or cfg_net.get('ping_test_port', 53)
        # Use timeout_s if provided, else from config, else default
        timeout_to_use = timeout_s if timeout_s is not None else cfg_net.get('ping_timeout_s', 3)
        try:
            s = socket.create_connection((host, port), timeout=timeout_to_use)
            s.close()
            return True
        except (socket.error, socket.timeout): return False

    def setup_routes(self):
        @self.app.route('/api/data-usage') # Example, needs SimMonitor integration (P30)
        def get_data_usage():
            # P30: Return cached SIM data usage report
            with self.sim_data_lock:
                if self.last_sim_data_usage_report:
                    return jsonify(self.last_sim_data_usage_report)
                else:
                    return jsonify({"message": "SIM data usage report not yet available."}), 404
        
        @self.app.route('/api/status')
        def get_status():
            return jsonify({
                "current_speed_kmh": self.current_speed_kmh,
                "connectivity": self.check_connectivity(),
                "active_processes": {pd['name']: pd['process_obj'].is_alive() for pd in self.active_processes_details}
            })

    # --- Process Target Methods ---
    def _gps_process_target(self, interval_pipe_r, data_q):
        # Note: self passed to Process target refers to the main process's VehicleTracker.
        # Config is copied. For sensors, better to pass config values or init within process.
        # Here, using self.config which is a copy in the new process.
        logger = logging.getLogger(f"{__name__}.GPSProcess") # Process-specific logger
        cfg = self.config['gps']
        sensor = GPS(**cfg)
        if not sensor.initialize():
            logger.error("Failed to init GPS sensor. Exiting process.")
            return

        current_interval = self.config['logging']['interval']['gps']
        last_poll = 0
        logger.info(f"Started with interval {current_interval}s.")
        while True:
            if interval_pipe_r.poll():
                try: current_interval = interval_pipe_r.recv()
                except EOFError: break # Pipe closed
                logger.info(f"New interval: {current_interval}s.")
            
            if time.time() - last_poll >= current_interval:
                try: data = sensor.get_data()
                except Exception as e: logger.error(f"Sensor error: {e}"); data = None
                if data: data_q.put(('gps', data))
                last_poll = time.time()
            time.sleep(max(0.001, min(0.1, current_interval / 10.0 if current_interval > 0 else 0.1)))
        sensor.close(); logger.info("Exiting.")


    def _imu_process_target(self, interval_pipe_r, data_q):
        logger = logging.getLogger(f"{__name__}.IMUProcess")
        cfg = self.config['imu']
        sensor = IMU(**cfg)
        if not sensor.initialize():
            logger.error("Failed to init IMU sensor. Exiting process.")
            return

        current_interval = self.config['logging']['interval']['imu']
        last_poll = 0
        logger.info(f"Started with interval {current_interval}s.")
        while True:
            if interval_pipe_r.poll():
                try: current_interval = interval_pipe_r.recv()
                except EOFError: break
                logger.info(f"New interval: {current_interval}s.")
            
            if time.time() - last_poll >= current_interval:
                try: data = sensor.read_data()
                except Exception as e: logger.error(f"Sensor error: {e}"); data = None
                if data: data_q.put(('imu', data))
                last_poll = time.time()
            time.sleep(max(0.001, min(0.01, current_interval / 10.0 if current_interval > 0 else 0.01)))
        sensor.close(); logger.info("Exiting.")

    def _camera_process_target(self, interval_pipe_r, data_q, main_config_path):
        logger = logging.getLogger(f"{__name__}.CameraProcess")
        cam_cfg = self.config['camera'] # This self.config is a copy in the new process
        
        local_cam = None
        local_detector = None
        cam_initialized = False
        
        # Initialize SignDetector locally in this process
        try:
            local_detector = SignDetector(main_config_path)
            logger.info("SignDetector initialized locally.")
        except Exception as e:
            logger.error(f"Failed to init SignDetector locally: {e}. No detections will occur.")
            return # Critical failure

        frame_q_for_detection = Queue(maxsize=cam_cfg.get('internal_frame_queue_size', 2))
        capture_stop_event = threading.Event()
        capture_thread = None

        def _init_cam_locally():
            nonlocal local_cam, cam_initialized
            if local_cam: local_cam.close()
            
            # Try configured device, then glob
            dev_id = cam_cfg.get('device_id')
            devices_to_try = [dev_id] if dev_id else []
            devices_to_try.extend(d for d in glob.glob("/dev/video*") if d not in devices_to_try)

            for dev_path in devices_to_try:
                if not dev_path: continue
                cam = Camera(dev_path, cam_cfg['width'], cam_cfg['height'], cam_cfg['fps'])
                if cam.initialize():
                    local_cam, cam_initialized = cam, True
                    logger.info(f"Camera initialized: {dev_path}")
                    return True
                else: cam.close()
            logger.error("Failed to initialize any camera.")
            local_cam, cam_initialized = None, False
            return False

        def _capture_thread_fn(cam, q, stop_ev):
            while not stop_ev.is_set():
                if cam and cam_initialized:
                    try: frame = cam.get_frame()
                    except Exception as e: logger.warning(f"Capture get_frame error: {e}"); frame = None; time.sleep(0.5)
                    
                    if frame is not None:
                        try: q.put(frame, timeout=0.05) # Small timeout
                        except QueueFull: pass # Drop frame
                    else: time.sleep(0.1) # No frame
                else: time.sleep(0.2) # Camera not ready
            logger.info("Capture thread stopped.")

        current_interval = self.config['logging']['interval']['camera']
        last_detection_time = last_cam_init_try = 0
        reinit_interval = cam_cfg.get('reinit_interval_s', 30)
        logger.info(f"Started with detection interval {current_interval}s.")

        if _init_cam_locally():
            capture_thread = threading.Thread(target=_capture_thread_fn, args=(local_cam, frame_q_for_detection, capture_stop_event), daemon=True)
            capture_thread.start()
        else: last_cam_init_try = time.time()

        while True:
            if interval_pipe_r.poll():
                try: current_interval = interval_pipe_r.recv()
                except EOFError: break
                logger.info(f"New detection interval: {current_interval}s.")

            now = time.time()
            if not cam_initialized and (now - last_cam_init_try >= reinit_interval):
                logger.info("Attempting camera re-initialization...")
                if capture_thread and capture_thread.is_alive():
                    capture_stop_event.set(); capture_thread.join(timeout=1.0)
                
                if _init_cam_locally():
                    capture_stop_event.clear()
                    capture_thread = threading.Thread(target=_capture_thread_fn, args=(local_cam, frame_q_for_detection, capture_stop_event), daemon=True)
                    capture_thread.start()
                last_cam_init_try = now
            
            if cam_initialized and local_detector and (now - last_detection_time >= current_interval):
                try:
                    frame = frame_q_for_detection.get(timeout=0.1)
                    frame_copy = frame.copy() # For detection if it modifies frame
                    signs = local_detector.detect(frame_copy)
                    if signs: data_q.put(('signs', signs, frame_copy))
                except QueueEmpty: pass
                except Exception as e: logger.error(f"Detection error: {e}")
                last_detection_time = now
            
            time.sleep(max(0.001, min(0.1, current_interval / 20.0 if current_interval > 0 else 0.1)))

        # Cleanup for camera_process
        if capture_thread and capture_thread.is_alive():
            capture_stop_event.set(); capture_thread.join(timeout=1.0)
        if local_cam: local_cam.close()
        if local_detector and hasattr(local_detector, 'close'): local_detector.close()
        logger.info("Exiting.")


    def _data_transmission_process_target(self, data_q):
        logger = logging.getLogger(f"{__name__}.DataTx")
        current_gps_payload = {} # Store last good GPS for enriching detections
        # Calculate speed within this process based on received GPS/IMU data
        # This avoids relying on self.current_speed_kmh from the main VehicleTracker instance,
        # as that might not be perfectly in sync if this process handles speed calculation for packets.
        # However, filter_duplicate_detections in the main instance uses self.current_speed_kmh.
        # For consistency, it might be better if filter_duplicate_detections is also called here
        # or if speed context is passed reliably. Current filter_duplicate_detections uses self.current_speed_kmh.
        # Let's assume for now that self.current_speed_kmh updated in main thread is sufficient context
        # for filter_duplicate_detections if called from main thread context, or speed is passed if called here.

        while True:
            try:
                data_type, payload_item = data_q.get(timeout=self.batch_time_limit_s)
                
                if data_type == 'gps' and payload_item: current_gps_payload = payload_item

                packet = {'timestamp': datetime.now().isoformat()}
                frame_for_packet = None

                if data_type == 'gps': packet['gps'] = payload_item
                elif data_type == 'imu': packet['imu'] = payload_item
                elif data_type == 'signs':
                    signs_list, frame_img = payload_item
                    if current_gps_payload: packet['gps'] = current_gps_payload # Enrich with last GPS
                    
                    # Filter signs based on this GPS context
                    # Note: self.current_speed_kmh isn't directly available here.
                    # filter_duplicate_detections should ideally get current speed if needed,
                    # or rely only on position and time. Or, speed could be part of current_gps_payload.
                    # For simplicity, passing current_gps_payload which now includes speed.
                    filtered = self.filter_duplicate_detections(signs_list, current_gps_payload, time.time())
                    if filtered:
                        packet['signs'] = filtered
                        frame_for_packet = frame_img
                elif data_type == 'sim': # sim_process sends dict directly
                    sim_payload = payload_item # Assuming payload_item is the dict from sim_process
                    # P30: Cache the data usage report part for the API
                    if 'data_usage_report' in sim_payload:
                        with self.sim_data_lock:
                            self.last_sim_data_usage_report = sim_payload['data_usage_report']
                            logger.debug(f"Cached SIM data usage report for API: {self.last_sim_data_usage_report}")
                    
                    loop = self._get_or_create_event_loop() 
                    if not loop.run_until_complete(self._send_to_backend_async(sim_payload, self.config['backend']['sim_data_endpoint'])):
                        self.logger.warning(f"Failed to send SIM data: {sim_payload}") 
                    continue # Skip batching for SIM data

                if packet.get('gps') or packet.get('imu') or packet.get('signs'):
                    self.batch_outgoing_data.append({'content': packet, 'frame': frame_for_packet})

            except QueueEmpty: # Timeout from data_q.get()
                pass # Proceed to check batch send conditions
            except Exception as e:
                logger.error(f"Error processing data queue: {e}", exc_info=True)

            # Send batch if conditions met or on timeout
            if self.batch_outgoing_data and \
               (len(self.batch_outgoing_data) >= self.batch_size_limit or \
                time.time() - self.last_batch_send_time >= self.batch_time_limit_s):
                
                for item_in_batch in self.batch_outgoing_data:
                    self._send_single_packet(item_in_batch['content'], item_in_batch['frame'])
                self.batch_outgoing_data = []
                self.last_batch_send_time = time.time()

            self.send_offline_data_batch() # Try to send any offline data
            self.adjust_intervals() # Adjust sensor intervals based on conditions


    def _monitor_processes_target(self):
        logger = logging.getLogger(f"{__name__}.Monitor")
        monitor_interval = self.process_monitor_interval_s # From config via self
        logger.info(f"Started. Checking every {monitor_interval}s.")

        while True:
            for i, p_detail in enumerate(list(self.active_processes_details)): # Iterate copy for safe modification
                current_p_obj = p_detail['process_obj']
                if not current_p_obj.is_alive():
                    logger.error(f"Process {p_detail['name']} (PID {current_p_obj.pid if current_p_obj.pid else 'N/A'}) died. Restarting...")
                    try:
                        new_p = Process(name=p_detail['name'], target=p_detail['target'], args=p_detail['args'], daemon=True)
                        new_p.start()
                        self.active_processes_details[i]['process_obj'] = new_p # Update object in list
                        logger.info(f"Process {new_p.name} restarted (PID {new_p.pid}).")
                    except Exception as e: logger.error(f"Failed to restart {p_detail['name']}: {e}")
            time.sleep(monitor_interval)

    def run(self):
        if not self.initialize_main_components():
            self.logger.warning("Initial component checks had issues. Proceeding, but sub-processes might fail.")

        process_targets_args = [
            {'name': 'GPSProcess', 'target': self._gps_process_target, 'args': (self.gps_interval_pipe_recv, self.data_queue)},
            {'name': 'IMUProcess', 'target': self._imu_process_target, 'args': (self.imu_interval_pipe_recv, self.data_queue)},
            {'name': 'CameraProcess', 'target': self._camera_process_target, 'args': (self.camera_interval_pipe_recv, self.data_queue, self.config_path)},
            {'name': 'DataTxProcess', 'target': self._data_transmission_process_target, 'args': (self.data_queue,)},
            {'name': 'SimProcess', 'target': sim_process, 'args': (self.config, self.data_queue)} # sim_process is external
        ]

        for pt_args in process_targets_args:
            try:
                p = Process(name=pt_args['name'], target=pt_args['target'], args=pt_args['args'], daemon=True)
                p.start()
                self.active_processes_details.append({**pt_args, 'process_obj': p}) # Store full detail
                self.logger.info(f"Started process {p.name} (PID {p.pid}).")
            except Exception as e: self.logger.error(f"Failed to start {pt_args['name']}: {e}", exc_info=True)

        api_cfg = self.config['api']
        flask_host = api_cfg.get('host','0.0.0.0')
        flask_port = api_cfg.get('port', 5000)
        flask_thread = threading.Thread(target=self.app.run, 
                                        kwargs={'host': flask_host, 
                                                'port': flask_port, 
                                                'use_reloader': False}, daemon=True)
        flask_thread.start()
        self.logger.info(f"Flask API server starting on http://{flask_host}:{flask_port}")

        if self.active_processes_details:
            monitor_thread = threading.Thread(target=self._monitor_processes_target, daemon=True)
            monitor_thread.start()
        else: self.logger.warning("No processes started; monitor thread not initiated.")

        try:
            while True: time.sleep(1.0) # Main thread keeps alive
        except KeyboardInterrupt: self.logger.info("KeyboardInterrupt received.")
        finally:
            self.logger.info("Shutting down...")
            # Terminate processes
            for p_detail in self.active_processes_details:
                try:
                    if p_detail['process_obj'].is_alive():
                        self.logger.info(f"Terminating {p_detail['name']}...")
                        p_detail['process_obj'].terminate()
                        p_detail['process_obj'].join(timeout=2.0) # Wait for termination
                        if p_detail['process_obj'].is_alive():
                             self.logger.warning(f"{p_detail['name']} did not terminate cleanly.")
                except Exception as e: self.logger.error(f"Error terminating {p_detail['name']}: {e}")
            
            self.cleanup_resources()
            self.logger.info("Shutdown complete.")

    def cleanup_resources(self):
        self.logger.info("Cleaning up main VehicleTracker resources...")
        # Close pipe send ends
        for pipe_send_end in [self.gps_interval_pipe_send, self.imu_interval_pipe_send, self.camera_interval_pipe_send]:
            if pipe_send_end:
                try: pipe_send_end.close()
                except Exception as e: self.logger.debug(f"Pipe close error: {e}")
        
        # Other resources like DB engine, if explicitly managed (SQLAlchemy handles it mostly)
        if self.engine:
            try: self.engine.dispose()
            except Exception as e: self.logger.debug(f"DB engine dispose error: {e}")

        self.logger.info("Main resources cleanup finished.")


if __name__ == "__main__":
    cfg_path = "config/config.yaml"
    # Basic logging before VehicleTracker's full setup, if needed for early diags
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    main_tracker = VehicleTracker(config_path=cfg_path)
    main_tracker.run()