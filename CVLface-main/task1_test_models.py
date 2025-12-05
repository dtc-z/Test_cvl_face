import os
import sys
import numpy as np
import torch
import argparse
import inspect
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cvlface"))

np.bool = np.bool_
np.object = np.object_
np.float = np.float_

from general_utils.huggingface_model_utils import load_model_by_repo_id
from general_utils.img_utils import visualize, prepare_text_img

MODELS_DIR = os.path.join(ROOT, "cvlface", "pretrained_models", "recognition")
MODELS = [
    "adaface_ir101_webface4m",
    "arcface_ir101_webface4m",
    "adaface_ir101_webface12m",
]
ALIGNER_ID = "minchul/cvlface_DFA_mobilenet"
DEFAULT_THRESHOLD = 0.3


def pil_to_input(pil_image, device):
    trans = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return trans(pil_image).unsqueeze(0).to(device)


def load_local_model(model_name, device):
    model_path = os.path.join(MODELS_DIR, model_name)
    config_path = os.path.join(model_path, "config.yaml")
    model_pt_path = os.path.join(model_path, "model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_pt_path):
        print(f"[{model_name}] Missing config or model.pt")
        return None

    print(f"[{model_name}] Loading from {model_path}...")
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    try:
        loaded = torch.load(model_pt_path, map_location=device)
        
        if isinstance(loaded, dict):
            print(f"[{model_name}] Loaded dict (likely state_dict)")
            if "model" in loaded:
                model = loaded["model"]
            elif "state_dict" in loaded:
                state_dict = loaded["state_dict"]
                from cvlface.research.face_recognition.model_zoo import get_model
                model = get_model(cfg)
                model.load_state_dict(state_dict, strict=False)
            else:
                state_dict = loaded
                from cvlface.research.face_recognition.model_zoo import get_model
                model = get_model(cfg)
                model.load_state_dict(state_dict, strict=False)
        else:
            model = loaded
        
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"[{model_name}] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_model(model, aligner, img1_path, img2_path, device, threshold):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    inp1 = pil_to_input(img1, device)
    inp2 = pil_to_input(img2, device)

    with torch.no_grad():
        aligned_x1, _, aligned_ldmks1, _, _, _ = aligner(inp1)
        aligned_x2, _, aligned_ldmks2, _, _, _ = aligner(inp2)

        sig = inspect.signature(model.forward)
        if sig.parameters.get("keypoints") is not None:
            feat1 = model(aligned_x1, aligned_ldmks1)
            feat2 = model(aligned_x2, aligned_ldmks2)
        else:
            feat1 = model(aligned_x1)
            feat2 = model(aligned_x2)

        cossim = torch.nn.functional.cosine_similarity(feat1, feat2).item()

    is_same = cossim > threshold
    return cossim, is_same


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CVLface local models")
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading aligner model...")
    aligner = load_model_by_repo_id(
        repo_id=ALIGNER_ID,
        save_path=os.path.expanduser(f"~/.cvlface_cache/{ALIGNER_ID}"),
        HF_TOKEN=os.environ.get("HF_TOKEN", ""),
    ).to(device)
    aligner.eval()

    print("\nTesting models:\n")
    results = {}

    for model_name in MODELS:
        print("=" * 60)
        print(f"Model: {model_name}")
        print("=" * 60)

        model = load_local_model(model_name, device)
        if model is None:
            results[model_name] = None
            print()
            continue

        cossim, is_same = compare_with_model(
            model, aligner, args.img1, args.img2, device, args.threshold
        )

        print(f"Similarity: {cossim:.4f}")
        print(f"Threshold: {args.threshold:.2f}")
        print("Result:", "SAME PERSON" if is_same else "DIFFERENT PERSON")
        print()

        results[model_name] = (cossim, is_same)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, res in results.items():
        if res is None:
            print(f"{model_name:30s}: FAILED")
        else:
            cossim, is_same = res
            label = "SAME" if is_same else "DIFF"
            print(f"{model_name:30s}: {cossim:.4f} ({label})")
