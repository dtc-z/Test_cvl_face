import os
import struct
import random
import shutil
from tqdm import tqdm
import cv2
import numpy as np

REC_PATH = "data/faces_webface_112x112/train.rec"
IDX_PATH = "data/faces_webface_112x112/train.idx"
OUTPUT_DIR = "data/my_1000_id_dataset_hq"
TARGET_IDS = 1000
QUALITY_THRESHOLD = 270
IMGS_PER_ID = (15, 15)
MIN_HIGH_QUALITY_IMGS = 15  # Must have at least 15 images with quality >= avg

def read_idx(idx_path):
    index = {}
    with open(idx_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            k, v = parts
            index[int(k)] = int(v)
    return index

def read_record_header(rec_f, pos):
    rec_f.seek(pos)
    data = rec_f.read(16)
    if len(data) != 16:
        return None
    magic, img_len, flag, label_float = struct.unpack('IIIf', data)
    return int(label_float)

def read_full_record(rec_f, pos):
    rec_f.seek(pos)
    header = rec_f.read(8)
    magic, img_len = struct.unpack('II', header)
    rec_header = rec_f.read(24)
    img_bytes = rec_f.read(img_len)
    return img_bytes

def get_image_quality(img_bytes):
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def main():
    if not os.path.exists(REC_PATH) or not os.path.exists(IDX_PATH):
        print(f"Error: cannot find {REC_PATH} or {IDX_PATH}")
        return

    print("Reading index...")
    idx_map = read_idx(IDX_PATH)
    max_idx = max(idx_map.keys())

    print("Scanning records and computing quality per ID...")
    id_to_images = {}
    
    with open(REC_PATH, 'rb') as f:
        for i in tqdm(range(1, max_idx + 1)):
            if i not in idx_map:
                continue
            pos = idx_map[i]
            label = read_record_header(f, pos)
            if label is None:
                continue
            
            img_bytes = read_full_record(f, pos)
            quality = get_image_quality(img_bytes)
            
            if label not in id_to_images:
                id_to_images[label] = []
            id_to_images[label].append({
                'rec_idx': i,
                'quality': quality,
                'img_bytes': img_bytes
            })

    print(f"Found {len(id_to_images)} unique identities.")
    
    # Filter IDs by:
    # 1. Average quality > QUALITY_THRESHOLD
    # 2. At least MIN_HIGH_QUALITY_IMGS images with quality >= average
    valid_ids = []
    for id_label, images in id_to_images.items():
        avg_quality = sum(img['quality'] for img in images) / len(images)
        
        if avg_quality <= QUALITY_THRESHOLD:
            continue
        
        # Count how many images have quality >= average
        high_quality_count = len([img for img in images if img['quality'] >= avg_quality])
        
        if high_quality_count >= MIN_HIGH_QUALITY_IMGS:
            valid_ids.append((id_label, avg_quality, high_quality_count))
    
    print(f"Found {len(valid_ids)} IDs with avg quality > {QUALITY_THRESHOLD} and at least {MIN_HIGH_QUALITY_IMGS} high-quality images")
    
    if len(valid_ids) < TARGET_IDS:
        print(f"Warning: only {len(valid_ids)} valid IDs. Using all.")
        ids_selected = [id_label for id_label, _, _ in valid_ids]
    else:
        # Sort by average quality (descending) and pick top TARGET_IDS
        valid_ids.sort(key=lambda x: x[1], reverse=True)
        ids_selected = [id_label for id_label, _, _ in valid_ids[:TARGET_IDS]]

    print(f"Selected {len(ids_selected)} IDs for extraction.")
    
    if os.path.exists(OUTPUT_DIR):
        print(f"Cleaning existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Extracting high-quality images...")
    for idx, id_label in enumerate(tqdm(ids_selected)):
        images = id_to_images[id_label]
        
        # Compute avg quality for this ID
        avg_quality = sum(img['quality'] for img in images) / len(images)
        
        # Filter images: quality >= avg_quality
        high_quality_images = [img for img in images if img['quality'] >= avg_quality]
        
        # Sort by quality (descending)
        high_quality_images.sort(key=lambda x: x['quality'], reverse=True)
        
        # Pick exactly between 15-20 images (guaranteed to have at least 15)
        count = random.randint(*IMGS_PER_ID)
        count = min(count, len(high_quality_images))
        selected_images = high_quality_images[:count]
        
        # Create folder
        folder_name = f"ID_{idx+1:03d}"
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # Save images
        for j, img_data in enumerate(selected_images):
            img_bytes = img_data['img_bytes']
            
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
            
            fname = os.path.join(folder_path, f"{folder_name}_img{j+1}.jpg")
            cv2.imwrite(fname, img)

    print(f"\nDone!")
    print(f"Created {len(ids_selected)} IDs")
    print(f"Each ID has 15-20 images with quality >= their average")
    print(f"Location: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()