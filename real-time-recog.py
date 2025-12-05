import os
import sys
import cv2
import numpy as np
import json
import time
import argparse
import torch
# import importlib.util
from pathlib import Path
from collections import defaultdict
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import inspect
# from transformers import AutoModel
from cvlface_align_loader import load_dfa_aligner, load_cvlface_model_hf

# ---------------- Paths / config ----------------

INSIGHTFACE_ROOT = os.path.join(os.getcwd(), "insightface-master", "python-package")
sys.path.insert(0, INSIGHTFACE_ROOT)

from insightface.model_zoo import SCRFD

MODEL_PATH_SCRFD = ".insightface_cache/models/custom_scrfd/scrfd_2.5g_bnkps.onnx"
CACHE_DIR = ".cvlface_local_cache"

CONFIDENCE_THRESHOLD = 0.5 # for face detection
INPUT_RESOLUTION = (640, 480)
DETECTION_FPS = 10
DISPLAY_FPS = 30
PADDING_RATIO = 0.1  # small padding around bbox to match training scale

RECOGNITION_THRESHOLD = 0.25  # use threshold from your eval
ALIGNED_FACE_SIZE = 112
EMBEDDING_DIM = 512

REGISTERED_DB_DIR = os.path.abspath("database/registered_database_ada_12m_x") # REMEMBER TO EDIT THIS BEFORE RUNNING
REGISTERED_EMBEDDINGS = os.path.join(REGISTERED_DB_DIR, "embeddings.npy")
REGISTERED_METADATA = os.path.join(REGISTERED_DB_DIR, "metadata.json")

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

MODELS = [
    "minchul_cvlface_adaface_ir101_webface4m",
    "minchul_cvlface_arcface_ir101_webface4m",
    "minchul_cvlface_adaface_ir101_webface12m",
    # "minchul_cvlface_adaface_vit_base_kprpe_webface4m",
    "minchul_cvlface_adaface_vit_base_webface4m",
]

MODEL_ID = MODELS[2] # REMEMBER TO EDIT THIS BEFORE RUNNING

ALIGNER_ID = "minchul/cvlface_DFA_mobilenet"

# ---------------- SCRFD detector ----------------

class SCRFDDetector:
    def __init__(self, model_path=MODEL_PATH_SCRFD, device="cuda"):
        self.device = device
        self.confidence_threshold = CONFIDENCE_THRESHOLD

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        print(f"Loading SCRFD: {model_path}...")
        try:
            self.detector = SCRFD(model_file=model_path)
            ctx_id = 0 if device == "cuda" else -1
            self.detector.prepare(ctx_id=ctx_id, input_size=INPUT_RESOLUTION)
            print("✓ SCRFD loaded\n")
        except Exception as e:
            print(f"Error: {e}")
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
        bboxes, kpss = self.detector.detect(frame, INPUT_RESOLUTION)

        results = []
        h, w = frame.shape[:2]

        if bboxes is not None and len(bboxes) > 0:
            for idx, bbox_raw in enumerate(bboxes):
                score = float(bbox_raw[4])
                if score < self.confidence_threshold:
                    continue

                bbox = bbox_raw[:4].astype(int)
                bbox_padded = self.add_padding(bbox, h, w)

                results.append({
                    "bbox": bbox,
                    "bbox_padded": bbox_padded,
                    "confidence": score,
                    "kps": kpss[idx] if kpss is not None else None
                })

        return results

# ---------------- Face recognizer ----------------

class CVLFaceRecognizer:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading CVLFace model and DFA aligner...")
        self.aligner = load_dfa_aligner("minchul/cvlface_DFA_mobilenet", device)
        self.model = load_cvlface_model_hf(MODEL_ID, device)
        print("✓ CVLFace model + aligner loaded\n")

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _crop_to_pil(self, frame_bgr, bbox_padded):
        x1, y1, x2, y2 = bbox_padded
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        return pil_img

    def get_embedding(self, frame_bgr, bbox_padded):
        """Match face_verify_img.py: crop → aligner → model → L2 norm."""
        try:
            pil_img = self._crop_to_pil(frame_bgr, bbox_padded)
            if pil_img is None:
                return None

            inp = self.transform(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                aligned, _, ldmks, _, _, _ = self.aligner(inp)

                sig = inspect.signature(self.model.forward)
                if sig.parameters.get("keypoints") is not None:
                    feat = self.model(aligned, ldmks)
                else:
                    feat = self.model(aligned)

                feat = feat / torch.norm(feat, dim=1, keepdim=True)

            embedding = feat.cpu().numpy().flatten().astype(np.float32)
            return embedding

        except Exception as e:
            print(f"Error getting embedding: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def compute_similarity(embedding1, embedding2):
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0

            embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
            embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
            similarity = float(np.dot(embedding1, embedding2))
            return similarity
        except Exception:
            return 0.0

# ---------------- Database ----------------

class FaceDatabase:
    def __init__(self):
        self.data = self.load()

    def load(self):
        if not os.path.exists(REGISTERED_EMBEDDINGS):
            print(f"Warning: Database not found at {REGISTERED_EMBEDDINGS}")
            return {"faces": {}, "threshold": RECOGNITION_THRESHOLD}

        if not os.path.exists(REGISTERED_METADATA):
            print(f"Warning: Metadata not found at {REGISTERED_METADATA}")
            return {"faces": {}, "threshold": RECOGNITION_THRESHOLD}

        try:
            embeddings = np.load(REGISTERED_EMBEDDINGS)
            print(f"✓ Loaded embeddings: {embeddings.shape}")

            with open(REGISTERED_METADATA, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Loaded metadata: {len(metadata)} entries\n")

            identity_embeddings = defaultdict(list)
            for entry in metadata:
                identity_id = entry["identity_id"]
                embedding_idx = entry["embedding_idx"]
                embedding = embeddings[embedding_idx]
                identity_embeddings[identity_id].append(embedding)

            faces_dict = {}
            for identity_id, emb_list in sorted(identity_embeddings.items()):
                avg_embedding = np.mean(emb_list, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                faces_dict[identity_id] = {
                    "embedding": avg_embedding.astype(np.float32),
                    "num_samples": len(emb_list)
                }

            return {
                "faces": faces_dict,
                "threshold": RECOGNITION_THRESHOLD
            }

        except Exception as e:
            print(f"Error loading database: {e}")
            import traceback
            traceback.print_exc()
            return {"faces": {}, "threshold": RECOGNITION_THRESHOLD}

    def get_faces(self):
        return self.data.get("faces", {})

    def get_threshold(self):
        return self.data.get("threshold", RECOGNITION_THRESHOLD)

# ---------------- Recognition helpers ----------------

def recognize_face(embedding, database, recognizer):
    db_faces = database.get_faces()
    threshold = database.get_threshold()

    if not db_faces or embedding is None:
        return "Unknown", 0.0, None

    max_similarity = 0.0
    best_match = "Unknown"
    all_similarities = {}

    start = time.time()

    for name, face_data in db_faces.items():
        known_embedding = face_data["embedding"]
        if isinstance(known_embedding, list):
            known_embedding = np.array(known_embedding, dtype=np.float32)

        similarity = recognizer.compute_similarity(embedding, known_embedding)
        all_similarities[name] = similarity

        if similarity > max_similarity:
            max_similarity = similarity
            if similarity > threshold:
                best_match = name

    elapsed = time.time() - start

    return best_match, max_similarity, all_similarities, elapsed

def draw_recognition_results(frame, detections, recognizer, database, embeddings):
    out = frame.copy()

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox_padded"]
        conf = det["confidence"]

        cv2.rectangle(out, (x1, y1), (x2, y2), BLUE, 2)

        if idx < len(embeddings) and embeddings[idx] is not None:
            name, max_similarity, _, elapsed = recognize_face(embeddings[idx], database, recognizer)
            text_color = GREEN if name != "Unknown" else RED
            text = f"{name} ({max_similarity:.2f}, {elapsed*1000:.1f} ms)"
        else:
            text_color = (200, 200, 200)
            text = f"Det: {conf:.2f}"

        tx, ty = x1, max(y1 - 10, 20)
        cv2.putText(out, text, (tx, ty), FONT, 0.6, text_color, 2)

    return out

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=RECOGNITION_THRESHOLD)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading database...")
    database = FaceDatabase()

    try:
        detector = SCRFDDetector(device=device)
        recognizer = CVLFaceRecognizer(device=device)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    print(f"Database: {REGISTERED_DB_DIR}")
    print(f"Loaded {len(database.get_faces())} identities")
    print(f"Recognition threshold: {args.threshold}\n")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, DISPLAY_FPS)

    frame_skip = max(1, int(DISPLAY_FPS / DETECTION_FPS))
    print(f"Starting recognition... (Press 'q' to exit)\n")

    last_detections = []
    last_embeddings = []
    frame_count = 0

    import time
    fps_clock = time.time()
    detection_clock = time.time()
    fps_counter = 0
    detection_counter = 0
    current_fps = 0.0
    current_detection_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                last_detections = detector.detect(frame)
                last_embeddings = []

                for det in last_detections:
                    bbox_padded = det["bbox_padded"]
                    embedding = recognizer.get_embedding(frame, bbox_padded)
                    last_embeddings.append(embedding)
            
                        # Track detection FPS
            detection_counter += 1
            elapsed_detection = time.time() - detection_clock
            if elapsed_detection >= 1.0:
                current_detection_fps = detection_counter / elapsed_detection
                detection_counter = 0
                detection_clock = time.time()

            annotated = draw_recognition_results(frame, last_detections, recognizer, database, last_embeddings)

            cv2.putText(annotated, f"Faces: {len(last_detections)}",
                        (10, 30), FONT, 0.6, (0, 255, 0), 1)
            cv2.putText(annotated, "Press 'q' to exit",
                        (10, INPUT_RESOLUTION[1] - 10), FONT, 0.5, (150, 150, 150), 1)
            # Track display FPS
            fps_counter += 1
            elapsed = time.time() - fps_clock
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_clock = time.time()
            
            # Draw FPS on screen
            cv2.putText(annotated, f"Display FPS: {current_fps:.1f} | Detection FPS: {current_detection_fps:.1f}",
                        (10, 60), FONT, 0.6, (0, 255, 0), 1)

            cv2.imshow("CVLFace Recognition (DFA aligned)", annotated)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
