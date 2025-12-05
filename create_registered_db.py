import os
import sys
import json
import gc
import inspect

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import AutoModel

# Paths
ROOT = os.path.join(os.getcwd(), "CVLface-main")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "cvlface"))

np.bool = np.bool_
np.object = np.object_
np.float = np.float_

DATASET_DIR = os.path.abspath("data/my_1000_id_dataset_hq")
REGISTERED_DB_DIR = os.path.abspath("database/registered_database_ada_12m_x")
REGISTERED_EMBEDDINGS = os.path.join(REGISTERED_DB_DIR, "embeddings.npy")
REGISTERED_METADATA = os.path.join(REGISTERED_DB_DIR, "metadata.json")

LOCAL_CACHE_DIR = os.path.abspath(".cvlface_local_cache")

MODELS = [
    "minchul_cvlface_adaface_ir101_webface4m",
    "minchul_cvlface_arcface_ir101_webface4m",
    "minchul_cvlface_adaface_ir101_webface12m",
    # "minchul_cvlface_adaface_vit_base_kprpe_webface4m",
    "minchul_cvlface_adaface_vit_base_webface4m",
]

MODEL_ID = MODELS[2]         
ALIGNER_ID = "minchul_cvlface_DFA_mobilenet"


def pil_to_input(pil_image, device):
    trans = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return trans(pil_image).unsqueeze(0).to(device)


def load_from_local_cache(model_id, device):
    cache_folder_name = model_id.replace("/", "_")
    local_path = os.path.join(LOCAL_CACHE_DIR, cache_folder_name)

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Model cache not found at {local_path}")

    print(f"  Loading from cache: {local_path}")

    cwd = os.getcwd()
    os.chdir(local_path)
    sys.path.insert(0, local_path)

    try:
        model = AutoModel.from_pretrained(
            local_path,
            trust_remote_code=True,
            device_map=device
        )
    finally:
        os.chdir(cwd)
        sys.path.pop(0)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


def extract_embedding(model, aligner, image_path, device):
    try:
        img = Image.open(image_path).convert("RGB")
        inp = pil_to_input(img, device)

        with torch.no_grad():
            aligned, _, ldmks, _, _, _ = aligner(inp)

            sig = inspect.signature(model.forward)
            if sig.parameters.get("keypoints") is not None:
                feat = model(aligned, ldmks)
            else:
                feat = model(aligned)

            feat = feat / torch.norm(feat, dim=1, keepdim=True)

        return feat.cpu().numpy().flatten()
    except Exception as e:
        print(f"    Error extracting embedding from {image_path}: {e}")
        return None


def main():
    os.makedirs(REGISTERED_DB_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("Loading models...")
    aligner = load_from_local_cache(ALIGNER_ID, device)
    model = load_from_local_cache(MODEL_ID, device)
    print("Models loaded\n")

    all_embeddings = []
    metadata_list = []

    id_folders = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )

    print(f"Found {len(id_folders)} ID folders\n")

    for id_idx, id_folder in enumerate(id_folders):
        id_path = os.path.join(DATASET_DIR, id_folder)
        images = sorted(f for f in os.listdir(id_path) if f.lower().endswith(".jpg"))

        registered_images = images[:5]

        print(f"ID {id_folder}: Taking {len(registered_images)} images for registration")

        for img_idx, img_name in enumerate(registered_images):
            img_path = os.path.join(id_path, img_name)

            embedding = extract_embedding(model, aligner, img_path, device)
            if embedding is not None:
                all_embeddings.append(embedding)
                metadata_list.append({
                    "identity_id": id_folder,
                    "image_name": img_name,
                    "image_path": img_path,
                    "embedding_idx": len(all_embeddings) - 1
                })
                print(f"  ✓ {img_name}")
            else:
                print(f"  ✗ {img_name} (failed)")
        torch.cuda.empty_cache()
        gc.collect()

    """ Extra image for Minh's face """

    embedding = extract_embedding(model, aligner, "data/DSC02617.JPG", device)
    if embedding is not None:
                all_embeddings.append(embedding)
                metadata_list.append({
                    "identity_id": "Minh",
                    "image_name": "Minh",
                    "image_path": img_path,
                    "embedding_idx": len(all_embeddings) - 1
                })
                print(f"  ✓ {img_name}")
    else:
                print(f"  ✗ {img_name} (failed)")
    torch.cuda.empty_cache()
    gc.collect()

    embeddings_array = np.array(all_embeddings)
    print(f"\nSaving {len(all_embeddings)} embeddings to {REGISTERED_EMBEDDINGS}")
    np.save(REGISTERED_EMBEDDINGS, embeddings_array)

    print(f"Saving metadata to {REGISTERED_METADATA}")
    with open(REGISTERED_METADATA, "w") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\n✓ Registered database created!")
    print(f"  Total identities: {len(set(m['identity_id'] for m in metadata_list))}")
    print(f"  Total registered images: {len(metadata_list)}")
    print(f"  Embedding shape: {embeddings_array.shape}")


if __name__ == "__main__":
    main()
