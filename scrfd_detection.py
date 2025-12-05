import os
import sys
import cv2
import numpy as np
import time
import torch

# Setup InsightFace (local clone)
INSIGHTFACE_ROOT = os.path.join(os.getcwd(), "insightface-master", "python-package")
sys.path.insert(0, INSIGHTFACE_ROOT)

# Import specific model class
from insightface.model_zoo import SCRFD

# ============ Configuration ============
MODEL_PATH = ".insightface_cache/models/custom_scrfd/scrfd_2.5g_bnkps.onnx"

CONFIDENCE_THRESHOLD = 0.5
INPUT_RESOLUTION = (640, 480)
DETECTION_FPS = 10
DISPLAY_FPS = 60
PADDING_RATIO = 0.10  # padding

# Colors (BGR)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class SCRFDDetector:
    """Custom SCRFD detector using direct ONNX loading"""
    
    def __init__(self, model_path=MODEL_PATH, device="cuda"):
        self.device = device
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}.\n"
                "Please run the download script first."
            )
        
        print(f"Loading ONNX model: {model_path}...")
        
        try:
            self.detector = SCRFD(model_file=model_path)
            ctx_id = 0 if device == "cuda" else -1
            self.detector.prepare(ctx_id=ctx_id, input_size=INPUT_RESOLUTION)
            print("✓ SCRFD 2.5G loaded successfully\n")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    def add_padding(self, bbox, frame_height, frame_width, padding_ratio=PADDING_RATIO):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        pad_x = int(width * padding_ratio)
        pad_y = int(height * padding_ratio)
        
        x1_pad = max(0, x1 - pad_x)
        y1_pad = max(0, y1 - pad_y)
        x2_pad = min(frame_width, x2 + pad_x)
        y2_pad = min(frame_height, y2 + pad_y)
        
        return np.array([x1_pad, y1_pad, x2_pad, y2_pad], dtype=int)

    def detect(self, frame):
        """Run detection - pass ONLY frame and threshold (input_size already set in prepare)"""
        # SCRFD.detect(frame, threshold) - that's it!
        bboxes, kpss = self.detector.detect(frame, INPUT_RESOLUTION)
        
        results = []
        h, w = frame.shape[:2]
        
        if bboxes is not None and len(bboxes) > 0:
            for idx, bbox_raw in enumerate(bboxes):
                bbox = bbox_raw[:4].astype(int)
                score = float(bbox_raw[4])
                
                bbox_padded = self.add_padding(bbox, h, w)
                
                results.append({
                    "bbox": bbox,
                    "bbox_padded": bbox_padded,
                    "confidence": score,
                    "kps": kpss[idx] if kpss is not None else None
                })
                
        return results

    def draw_boxes(self, frame, detections):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox_padded"]
            conf = det["confidence"]
            
            # Draw tight box in BLUE
            cv2.rectangle(out, (x1, y1), (x2, y2), BLUE, 2)
            
            # Draw confidence label
            text = f"{conf:.2f}"
            tx, ty = x1, max(y1 - 5, 20)
            cv2.putText(out, text, (tx, ty), FONT, 0.5, (255, 255, 255), 1)
            
        return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        detector = SCRFDDetector(device=device)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, DISPLAY_FPS)
    
    frame_skip = max(1, int(DISPLAY_FPS / DETECTION_FPS))
    print(f"Frame skip: {frame_skip} ({DISPLAY_FPS} display / {DETECTION_FPS} detect)")
    print(f"Padding: {int(PADDING_RATIO * 100)}%\n")
    
    last_detections = []
    frame_count = 0
    
    print("Starting... Press 'q' to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect only on skip frames
            if frame_count % frame_skip == 0:
                last_detections = detector.detect(frame)
                
            # Always draw last known detections
            annotated = detector.draw_boxes(frame, last_detections)
            
            # Draw info
            cv2.putText(annotated, f"SCRFD 2.5G | {len(last_detections)} faces", (10, 30), FONT, 0.6, (0, 255, 0), 1)
            
            cv2.imshow("SCRFD 2.5G (Custom ONNX) - Press 'q' to exit", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
            
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()