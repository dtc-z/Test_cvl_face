import os
import csv
import random
from glob import glob

DATASET_DIR = "data/my_50_id_dataset_hq"
OUTPUT_CSV = "evaluation/evaluation_pairs_1500.csv"

NUM_SAME_PAIRS = 500 
NUM_DIFF_PAIRS = 1000

def main():
    # Collect all IDs and their actual images (full paths)
    id_to_images = {}
    all_images_flat = []  # List of (id_name, full_path)
    
    id_dirs = sorted([d for d in os.listdir(DATASET_DIR) 
                      if os.path.isdir(os.path.join(DATASET_DIR, d))])
    
    print(f"Found {len(id_dirs)} ID folders")
    
    for id_dir in id_dirs:
        id_path = os.path.join(DATASET_DIR, id_dir)
        images = sorted(glob(os.path.join(id_path, "*.jpg")))
        
        if len(images) >= 2:
            id_to_images[id_dir] = images
            for img in images:
                all_images_flat.append((id_dir, img))
            print(f"  {id_dir}: {len(images)} images")
    
    print(f"\nTotal: {len(id_to_images)} IDs with >= 2 images")
    print(f"Total images: {len(all_images_flat)}\n")
    
    pairs = []
    pair_idx = 0
    
    # Generate SAME-ID pairs (500)
    print(f"Generating {NUM_SAME_PAIRS} same-ID pairs...")
    same_generated = 0
    for id_name in sorted(id_to_images.keys()):
        if same_generated >= NUM_SAME_PAIRS:
            break
        
        images = id_to_images[id_name]
        # Create all combinations within this ID
        for i in range(len(images)):
            for j in range(i+1, len(images)):
                if same_generated >= NUM_SAME_PAIRS:
                    break
                
                img_a_full = images[i]
                img_b_full = images[j]
                img_a_name = os.path.basename(img_a_full)
                img_b_name = os.path.basename(img_b_full)
                
                # Verify files actually exist
                if os.path.exists(img_a_full) and os.path.exists(img_b_full):
                    pairs.append((pair_idx, img_a_name, img_b_name, 1))  # 1 = same
                    pair_idx += 1
                    same_generated += 1
            
            if same_generated >= NUM_SAME_PAIRS:
                break
    
    print(f"Generated {same_generated} same-ID pairs\n")
    
    # Generate DIFFERENT-ID pairs (1000)
    print(f"Generating {NUM_DIFF_PAIRS} different-ID pairs...")
    diff_generated = 0
    attempts = 0
    max_attempts = NUM_DIFF_PAIRS * 20
    
    while diff_generated < NUM_DIFF_PAIRS and attempts < max_attempts:
        attempts += 1
        (id1, p1), (id2, p2) = random.sample(all_images_flat, 2)
        
        if id1 != id2:
            img_a_name = os.path.basename(p1)
            img_b_name = os.path.basename(p2)
            
            # Verify files actually exist
            if os.path.exists(p1) and os.path.exists(p2):
                pairs.append((pair_idx, img_a_name, img_b_name, 0))  # 0 = different
                pair_idx += 1
                diff_generated += 1
    
    print(f"Generated {diff_generated} different-ID pairs\n")
    
    # Write to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "img_a", "img_b", "gt_is_same"])
        for idx, a, b, gt in pairs:
            writer.writerow([idx, a, b, gt])
    
    print(f"Saved {len(pairs)} pairs to {OUTPUT_CSV}")
    print(f"Breakdown: {same_generated} same-ID + {diff_generated} different-ID")

if __name__ == "__main__":
    main()