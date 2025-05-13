import yaml
import cv2
import numpy as np
import onnxruntime as ort
from rknn.api import RKNN
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
            self.rknn = RKNN()
            self.rknn.load_rknn(self.config['rknn_model_path'])
            self.rknn.init_runtime(target='rk3588')
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def preprocess(self, frame):
        # GPU-based letterbox and normalization
        mf = cl.mem_flags
        img_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=frame)
        out_shape = (self.imgsz, self.imgsz, 3)
        out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, np.prod(out_shape) * frame.itemsize)
        self.program.letterbox(self.queue, frame.shape, None, img_buf, out_buf)
        img = np.empty(out_shape, dtype=np.uint8)
        cl.enqueue_copy(self.queue, img, out_buf)
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, 0)

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
        img = self.preprocess(frame)
        if self.model_type == 'onnx':
            outputs = self.session.run(None, {self.input_name: img})[0]
        else:
            outputs = self.rknn.inference(inputs=[img])[0]
        results = self.postprocess(outputs)
        if self.draw_boxes:
            for det in results:
                x, y, w, h = map(int, det['box'])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['label']} {det['confidence']:.2f}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.logger.debug(f"Detection took {(datetime.now() - start_time).total_seconds()*1000:.1f}ms")
        return results

    def close(self):
        if self.model_type == 'rknn':
            self.rknn.release()
