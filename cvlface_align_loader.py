from huggingface_hub import snapshot_download
import os
import sys
from transformers import AutoModel

LOCAL_CACHE_DIR = os.path.abspath(".cvlface_local_cache")

def load_dfa_aligner(repo_id, device):
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    cache_folder_name = repo_id.replace("/", "_")
    repo_local_root = os.path.join(LOCAL_CACHE_DIR, cache_folder_name)

    # Download (or reuse) local snapshot
    snapshot_download(
        repo_id=repo_id,
        local_dir=repo_local_root,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN", "")
    )

    cwd = os.getcwd()
    os.chdir(repo_local_root)
    sys.path.insert(0, repo_local_root)

    try:
        model = AutoModel.from_pretrained(
            repo_local_root,
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

def load_cvlface_model_hf(repo_id, device):
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    cache_folder_name = repo_id.replace("/", "_")
    repo_local_root = os.path.join(LOCAL_CACHE_DIR, cache_folder_name)

    snapshot_download(
        repo_id=repo_id,
        local_dir=repo_local_root,
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN", "")
    )

    cwd = os.getcwd()
    os.chdir(repo_local_root)
    sys.path.insert(0, repo_local_root)

    try:
        model = AutoModel.from_pretrained(
            repo_local_root,
            trust_remote_code=True,
            device_map=device,
        )
    finally:
        os.chdir(cwd)
        sys.path.pop(0)

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model