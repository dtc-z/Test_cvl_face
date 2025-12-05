# Test_cvl_face
# Test hiệu quả nhận diện khuôn mặt của model CVL Face, kết hợp SCRFD để phát hiện và nhận diện khuôn mặt real-time.

# Environment
Python 3.10 (thấp hơn hay cao hơn đều lỗi)

pip install -r ./CVLface-main/requirements.txt

# Clone repo 
git clone https://github.com/dtc-z/Test_cvl_face.git

cd Test_cvl_face

# 1. Test CVL face
## 1.1 Tạo database
Sửa các config DATASET_DIR, REGISTERED_DB_DIR, MODEL_ID

python create_registered_db.py
### Note: 
Database được tạo với model nào thì chỉ dùng được cho model đó 
## 1.2 Dùng CVL face so sánh 2 bức ảnh từ data (clean data, đã detect và crop)
Sửa các config REGISTERED_DB_DIR, MODEL_ID, THRESHOLD và TEST_PATH

python face_verify_img.py
## 1.3 Đánh giá CVL face trên data clean
### Tạo list các cặp ảnh cùng và khác ID để test:
Sửa các config DATASET_DIR, OUTPUT_CSV, NUM_SAME_PAIRS và NUM_DIFF_PAIRS

python gen_pairs.py
### Đánh giá:
Sửa các config DATASET_DIR, PAIRS_CSV, OUTPUT_DIR và THRESHOLDS

python eval_models.py
### Note: 
Dùng cùng 1 DATASET_DIR với DATASET_DIR khi chạy gen_pairs.py

Có thể bị crash khi quá tải RAM/GPU, lúc này comment các model khác trong list MODELS lại, chỉ đánh giá từng model hoặc 2 models cùng lúc
### Optional (tạo file html show report):
Sửa các config OUTPUT_DIR và REPORT_FILE

python gen_report.py

# 2. Test SCRFD 
Sửa các config CONFIDENCE_THRESHOLD, INPUT_RESOLUTION, DETECTION_FPS, DISPLAY_FPS và PADDING_RATIO

python scrfd_detection.py

# 3. Kết hợp SCRFD vào CVL face để phát hiện và nhận diện khuôn mặt real-time
Sửa các config CONFIDENCE_THRESHOLD, INPUT_RESOLUTION, DETECTION_FPS, DISPLAY_FPS, PADDING_RATIO, RECOGNITION_THRESHOLD, REGISTERED_DB_DIR và MODEL_ID

python real-time-recog.py
### Note:
Config đúng database và model, database được tạo với model nào thì chỉ dùng được cho model đó.
