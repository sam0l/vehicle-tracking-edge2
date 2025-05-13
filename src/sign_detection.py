import yaml
import cv2
import numpy as np
import onnxruntime as ort
from rknnlite.api import RKNNLite
import pyopencl as cl
import logging
from datetime import datetime

class SignDetector:
    def __init__(self, config_path):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['yolo']
        self.model_type = self.config['model_type']
        self.confidence_threshold = self.config['confidence_threshold']
        self.iou_threshold = self.config['iou_threshold']
        self.imgsz = self.config['imgsz']
        self.class_names = self.config['class_names']
        self.draw_boxes = self.config['draw_boxes']
        
        # Initialize OpenCL for GPU preprocessing
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        with open('preprocess.cl', 'r') as f:
            self.program = cl.Program(self.context, f.read()).build()
        
        # Initialize model
        if self.model_type == 'onnx':
            self.session = ort.InferenceSession(
                self.config['onnx_model_path'],
                providers=['CPUExecutionProvider'],
                provider_options=[{'intra_op_num_threads': self.config['intra_op_num_threads']}]
            )
            self.input_name = self.session.get_inputs()[0].name
        elif self.model_type == 'rknn':
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(self.config['rknn_model_path'])
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model: {self.config['rknn_model_path']}")
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise RuntimeError("Failed to initialize RKNN runtime")
            self.logger.info(f"RKNNLite initialized for {self.config['rknn_model_path']}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess(self, frame):
        # GPU-based letterbox and normalization
        mf = cl.mem_flags
        
        input_h, input_w = frame.shape[:2]
        
        # Calculate scale and padding for letterboxing (mirrors OpenCL logic)
        scale_w_py = self.imgsz / input_w
        scale_h_py = self.imgsz / input_h
        scale_py = min(scale_w_py, scale_h_py)
        
        scaled_input_w_py = int(input_w * scale_py)
        scaled_input_h_py = int(input_h * scale_py)
        
        pad_x_py = (self.imgsz - scaled_input_w_py) // 2
        pad_y_py = (self.imgsz - scaled_input_h_py) // 2

        img_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frame)
        # Output shape is (self.imgsz, self.imgsz, 3)
        # The OpenCL kernel handles 3 channels internally for each (x,y) pixel.
        # So, out_buf size is for self.imgsz * self.imgsz * 3 elements of frame.itemsize
        out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, self.imgsz * self.imgsz * 3 * frame.itemsize)
        
        # Kernel expects input_w, input_h, target_size
        # Global work size should be (target_size, target_size) for 2D, or (target_size, target_size, 1)
        # The kernel uses get_global_id(0) for out_x and get_global_id(1) for out_y.
        self.program.letterbox(self.queue, (self.imgsz, self.imgsz), None, 
                               img_buf, out_buf, 
                               np.int32(input_w), np.int32(input_h), np.int32(self.imgsz))
        
        # Create an empty array for the output from OpenCL (HWC format)
        letterboxed_img_hwc = np.empty((self.imgsz, self.imgsz, 3), dtype=frame.dtype)
        cl.enqueue_copy(self.queue, letterboxed_img_hwc, out_buf).wait() # Ensure copy is complete
        
        # Transpose to CHW, convert to float32, and normalize
        img_chw_float = letterboxed_img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Return processed image and scaling/padding info
        return np.expand_dims(img_chw_float, 0), scale_py, pad_x_py, pad_y_py

    def postprocess(self, outputs):
        # Simplified postprocessing (adjust based on YOLOv8 output format)
        detections = outputs[0]
        boxes, scores, classes = [], [], []
        for det in detections:
            if det[4] > self.confidence_threshold:
                boxes.append(det[:4])
                scores.append(det[4])
                classes.append(int(det[5]))
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)
        results = []
        for i in indices:
            box = boxes[i]
            results.append({
                'label': self.class_names[classes[i]],
                'confidence': scores[i],
                'box': box.tolist()
            })
        return results

    def detect(self, frame):
        start_time = datetime.now()
        # Get preprocessed image and scaling/padding info
        img, scale, pad_x, pad_y = self.preprocess(frame)
        
        if self.model_type == 'onnx':
            outputs = self.session.run(None, {self.input_name: img})[0]
        else:
            outputs = self.rknn.inference(inputs=[img])[0]
        results = self.postprocess(outputs)
        if self.draw_boxes:
            for det in results:
                # Box coordinates are for the letterboxed image (imgsz x imgsz)
                # Convert them back to original frame coordinates
                box_letterbox = det['box'] # Assuming [x, y, w, h] or [x1, y1, x2, y2]
                
                # Assuming box_letterbox format is [x_min, y_min, width, height]
                # If it's [x1,y1,x2,y2], adjust calculation for w,h or use x2,y2 directly
                lb_x, lb_y, lb_w, lb_h = box_letterbox
                
                # Map back to original frame
                orig_x = int((lb_x - pad_x) / scale)
                orig_y = int((lb_y - pad_y) / scale)
                orig_w = int(lb_w / scale)
                orig_h = int(lb_h / scale)
                
                # Draw on the original frame
                cv2.rectangle(frame, (orig_x, orig_y), (orig_x + orig_w, orig_y + orig_h), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['label']} {det['confidence']:.2f}", (orig_x, orig_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.logger.debug(f"Detection took {(datetime.now() - start_time).total_seconds()*1000:.1f}ms")
        return results

    def close(self):
        if self.model_type == 'rknn':
            self.rknn.release()
            self.logger.info("RKNNLite resources released")