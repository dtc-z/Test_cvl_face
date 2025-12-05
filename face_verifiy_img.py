import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import inspect
# import gc

ROOT = os.path.join(os.getcwd(), "CVLface-main")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cvlface"))

np.bool = np.bool_
np.object = np.object_
np.float = np.float_

from transformers import AutoModel

REGISTERED_DB_DIR = os.path.abspath("database/registered_database")
REGISTERED_EMBEDDINGS = os.path.join(REGISTERED_DB_DIR, "embeddings.npy")
REGISTERED_METADATA = os.path.join(REGISTERED_DB_DIR, "metadata.json")

# Local cache directory (already downloaded)
LOCAL_CACHE_DIR = os.path.abspath(".cvlface_local_cache")

# All 5 models
MODELS = [
    "minchul/cvlface_adaface_ir101_webface4m",
    "minchul/cvlface_arcface_ir101_webface4m",
    "minchul/cvlface_adaface_ir101_webface12m",
    "minchul/cvlface_adaface_vit_base_kprpe_webface4m",
    "minchul/cvlface_adaface_vit_base_webface4m",
]

# Use the best model from Step 1 evaluation
MODEL_ID = MODELS[2]  # cvlface_adaface_ir101_webface12m
ALIGNER_ID = "minchul/cvlface_DFA_mobilenet"
THRESHOLD = 0.24  # Adjust based on Step 1 evaluation results
TEST_PATH = ".\data\my_200_id_dataset_hq\ID_131\ID_131_img4.jpg"  # Replace with actual test image path

class FaceVerifier:
    def __init__(self, device="cuda"):
        self.device = device
        print("Loading models...")
        self.aligner = self._load_model(ALIGNER_ID)
        self.model = self._load_model(MODEL_ID)
        
        print("Loading registered database...")
        self.registered_embeddings = np.load(REGISTERED_EMBEDDINGS)
        with open(REGISTERED_METADATA) as f:
            self.metadata = json.load(f)
        
        print(f"✓ Initialized with {len(self.metadata)} registered images\n")
    
    def _load_model(self, model_id):
        """Load model from local cache"""
        cache_folder_name = model_id.replace("/", "_")
        local_path = os.path.join(LOCAL_CACHE_DIR, cache_folder_name)
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Model cache not found at {local_path}")
        
        cwd = os.getcwd()
        os.chdir(local_path)
        sys.path.insert(0, local_path)
        
        try:
            model = AutoModel.from_pretrained(
                local_path,
                trust_remote_code=True,
                device_map=self.device
            )
        except Exception as e:
            os.chdir(cwd)
            sys.path.pop(0)
            raise e
        
        os.chdir(cwd)
        sys.path.pop(0)
        
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    
    def _pil_to_input(self, pil_image):
        trans = Compose([
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        return trans(pil_image).unsqueeze(0).to(self.device)
    
    def extract_embedding(self, image_path):
        """Extract embedding from test image"""
        try:
            img = Image.open(image_path).convert("RGB")
            inp = self._pil_to_input(img)
            
            with torch.no_grad():
                aligned, _, ldmks, _, _, _ = self.aligner(inp)
                
                sig = inspect.signature(self.model.forward)
                if sig.parameters.get("keypoints") is not None:
                    feat = self.model(aligned, ldmks)
                else:
                    feat = self.model(aligned)
                
                # Normalize
                feat = feat / torch.norm(feat, dim=1, keepdim=True)
            
            return feat.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def verify(self, test_image_path):
        """
        Verify test image against registered database
        Returns: dict with best_match, confidence, threshold_passed, all_scores
        """
        test_embedding = self.extract_embedding(test_image_path)
        if test_embedding is None:
            return None
        
        # Compute cosine similarity with all registered embeddings
        similarities = np.dot(self.registered_embeddings, test_embedding)
        
        # Group by identity
        identity_scores = {}
        for idx, sim in enumerate(similarities):
            identity = self.metadata[idx]["identity_id"]
            if identity not in identity_scores:
                identity_scores[identity] = []
            identity_scores[identity].append(sim)
        
        # Get best score per identity (mean of top 3)
        identity_best_scores = {}
        for identity, scores in identity_scores.items():
            identity_best_scores[identity] = np.mean(sorted(scores, reverse=True)[:3])
        
        # Find best match
        best_identity = max(identity_best_scores, key=identity_best_scores.get)
        best_score = identity_best_scores[best_identity]
        
        return {
            "best_match": best_identity,
            "confidence": float(best_score),
            "threshold_passed": best_score >= THRESHOLD,
            "all_scores": {k: float(v) for k, v in identity_best_scores.items()}
        }

def main():
    verifier = FaceVerifier()
    
    # Example: verify a test image
    test_image = TEST_PATH  # Replace with actual path
    
    if os.path.exists(test_image):
        print(f"Verifying {test_image}...")
        result = verifier.verify(test_image)
        
        if result:
            print(f"\nResult:")
            print(f"  Best match: {result['best_match']}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Threshold ({THRESHOLD}): {'PASS' if result['threshold_passed'] else 'FAIL'}")
            print(f"\n  Top 5 candidates:")
            for idx, (identity, score) in enumerate(sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True)[:5], 1):
                print(f"    {idx}. {identity}: {score:.4f}")
    else:
        print(f"Test image not found: {test_image}")
        print("\nUsage:")
        print("  from step3_face_verification import FaceVerifier")
        print("  verifier = FaceVerifier()")
        print("  result = verifier.verify('path/to/image.jpg')")
        print(f"  print(f'Best match: {{result[\"best_match\"]}} (conf: {{result[\"confidence\"]:.4f}})')")

if __name__ == "__main__":
    main()