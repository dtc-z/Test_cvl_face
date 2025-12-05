import os
import sys
import csv
import time
import gc

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import inspect

from huggingface_hub import snapshot_download

# ---------- Paths & basic setup ----------

PROJECT_ROOT = os.getcwd()
CVLFACE_ROOT = os.path.join(PROJECT_ROOT, "CVLface-main")
sys.path.insert(0, CVLFACE_ROOT)
sys.path.insert(0, os.path.join(CVLFACE_ROOT, "cvlface"))

# Avoid old numpy aliases
np.bool = np.bool_
np.object = np.object_
np.float = np.float_

from general_utils.huggingface_model_utils import load_model_by_repo_id

DATASET_DIR = os.path.abspath("data/my_200_id_dataset_hq")
PAIRS_CSV   = "evaluation/evaluation_pairs_6000.csv"
OUTPUT_DIR  = os.path.abspath("evaluation_results_6000_pairs")

# Local cache under project 
LOCAL_CACHE_DIR = os.path.abspath(".cvlface_local_cache")

MODELS = [
    "minchul/cvlface_adaface_ir101_webface4m",
    "minchul/cvlface_arcface_ir101_webface4m",
    "minchul/cvlface_adaface_ir101_webface12m",
    # "minchul/cvlface_adaface_vit_base_kprpe_webface4m",
    "minchul/cvlface_adaface_vit_base_webface4m",
]

THRESHOLDS = [0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.4]
ALIGNER_ID = "minchul/cvlface_DFA_mobilenet"


# ---------- Helpers ----------

def pil_to_input(pil_image, device):
    trans = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return trans(pil_image).unsqueeze(0).to(device)


def load_from_hf_local(repo_id, device):
    """
    Download repo into project-local cache and load via CVLface helper.
    """
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    repo_local_root = os.path.join(LOCAL_CACHE_DIR, repo_id.replace("/", "_"))
    # Download once
    snapshot_download(
        repo_id=repo_id,
        local_dir=repo_local_root,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN", "")
    )

    # SPECIAL CASE: ViT-KPRPE model expects pretrained_model/model.pt
    if "cvlface_adaface_vit_base_kprpe_webface4m" in repo_id:
        import shutil

        pt_src = os.path.join(repo_local_root, "model.pt")
        pt_dst_dir = os.path.join(repo_local_root, "pretrained_model")
        pt_dst = os.path.join(pt_dst_dir, "model.pt")

        # Only copy if source exists and dest doesn't
        if os.path.exists(pt_src) and not os.path.exists(pt_dst):
            os.makedirs(pt_dst_dir, exist_ok=True)
            print(f"  Fixing path: copying {pt_src} -> {pt_dst}")
            shutil.copy2(pt_src, pt_dst)

    # Use CVLface's loader, but point save_path to our local cache
    print(f"  Loading {repo_id} from {repo_local_root} ...")
    model = load_model_by_repo_id(
        repo_id=repo_id,
        save_path=repo_local_root,
        HF_TOKEN=os.environ.get("HF_TOKEN", "")
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def build_image_map(dataset_dir):
    """filename -> full absolute path"""
    m = {}
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(".jpg"):
                m[f] = os.path.join(root, f)
    return m


def compare_pair(model, aligner, img_a_path, img_b_path, device):
    """Return cosine similarity or None if something fails."""
    try:
        if not (os.path.exists(img_a_path) and os.path.exists(img_b_path)):
            return None

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        inp_a = pil_to_input(img_a, device)
        inp_b = pil_to_input(img_b, device)

        with torch.no_grad():
            aligned_a, _, ldmk_a, _, _, _ = aligner(inp_a)
            aligned_b, _, ldmk_b, _, _, _ = aligner(inp_b)

            sig = inspect.signature(model.forward)
            if "keypoints" in sig.parameters:
                feat_a = model(aligned_a, ldmk_a)
                feat_b = model(aligned_b, ldmk_b)
            else:
                feat_a = model(aligned_a)
                feat_b = model(aligned_b)

            cossim = torch.nn.functional.cosine_similarity(feat_a, feat_b).item()

        del inp_a, inp_b, aligned_a, aligned_b, ldmk_a, ldmk_b, feat_a, feat_b
        return cossim
    except Exception:
        return None


# ---------- Main ----------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\nLocal cache: {LOCAL_CACHE_DIR}\n")

    # 1) Load pairs
    print("Loading pairs...")
    pairs = []
    with open(PAIRS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append({
                "idx": int(row["idx"]),
                "img_a": row["img_a"],
                "img_b": row["img_b"],
                "gt":  int(row["gt_is_same"]),
            })
    print(f"Loaded {len(pairs)} pairs\n")

    # 2) Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # 3) Load aligner (from local cache)
    print("Loading aligner...")
    aligner = load_from_hf_local(ALIGNER_ID, device)
    print("Aligner ready\n")

    # 4) Image map (once)
    print("Building image map...")
    image_map = build_image_map(DATASET_DIR)
    print(f"Found {len(image_map)} images\n")

    # 5) Evaluate each model
    for repo_id in MODELS:
        model_name = repo_id.split("/")[-1]
        print("=" * 60)
        print(f"Model: {model_name}")
        print("=" * 60)

        model = load_from_hf_local(repo_id, device)
        if model is None:
            print(f"Skipping {model_name}\n")
            continue

        print("Computing similarity scores...")
        scores_and_gts = []
        start = time.time()

        for i, pair in enumerate(pairs):
            a_name = pair["img_a"]
            b_name = pair["img_b"]

            if a_name not in image_map or b_name not in image_map:
                continue

            a_path = image_map[a_name]
            b_path = image_map[b_name]

            s = compare_pair(model, aligner, a_path, b_path, device)
            if s is not None:
                scores_and_gts.append((s, pair["gt"]))

            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(pairs)}")
                torch.cuda.empty_cache()
                gc.collect()

        if not scores_and_gts:
            print("ERROR: No pairs processed. Skipping.\n")
            del model
            torch.cuda.empty_cache()
            continue

        elapsed = time.time() - start
        avg_ms = (elapsed / len(scores_and_gts)) * 1000
        print(f"Processed {len(scores_and_gts)} pairs in {elapsed:.2f}s "
              f"(avg {avg_ms:.2f} ms/pair)")

        # Metrics per threshold
        results = []
        for thr in THRESHOLDS:
            pos_r = sum(1 for s, gt in scores_and_gts if gt == 1 and s >= thr)
            pos_w = sum(1 for s, gt in scores_and_gts if gt == 1 and s <  thr)
            neg_r = sum(1 for s, gt in scores_and_gts if gt == 0 and s <  thr)
            neg_w = sum(1 for s, gt in scores_and_gts if gt == 0 and s >= thr)

            results.append({
                "threshold":       thr,
                "avg_time_ms":     avg_ms,
                "positive_right":  pos_r,
                "positive_wrong":  pos_w,
                "negative_right":  neg_r,
                "negative_wrong":  neg_w,
            })

        out_csv = os.path.join(OUTPUT_DIR, f"results_{model_name}.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved to {out_csv}\n")

        del model, scores_and_gts
        torch.cuda.empty_cache()
        gc.collect()

    print("=" * 60)
    print(f"Done! Results in: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()