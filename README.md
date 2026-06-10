# Lab 6 — Computer Vision as IoT Sensor

> **Môn học**: Triển khai, phát triển ứng dụng AI và IoT  
> **Vị trí**: Buổi 6 trong chuỗi AIoT Deployment Pipeline  
> **Tác giả**: PHUPHU2310  
> **Cập nhật**: 2026-06-10

---

## Tổng quan

Lab này đưa camera hoặc ảnh vào hệ thống AIoT như một **cảm biến trực quan**. Thay vì đọc giá trị số từ cảm biến nhiệt độ hay độ ẩm, hệ thống đọc frame từ camera, trích xuất metadata, sinh visual event và hiển thị trên dashboard thời gian thực.

```
Camera / Ảnh
  → Stream / Snapshot / Upload
  → ROI crop (optional)
  → Xử lý ảnh (resize, grayscale, threshold, edge, mask, quality info)
  → Tính brightness + blur_score
  → Ghi metadata  → image_metadata.csv
  → Sinh event    → image_event_log.csv
  → Ghi tham số  → parameter_experiment_log.csv
  → 🔔 Motion Notification (banner + toast + âm thanh)
  → Dashboard (http://127.0.0.1:8000 | http://127.0.0.1:8001)
```

---

## Cấu trúc repository

```
lab6_cv_as_iot_ready_code/
├── lab6_cv_as_iot_sensor/              # Lab 6 cơ bản
│   ├── app.py                          # Backend FastAPI
│   ├── index.html                      # Dashboard
│   ├── run_lab6_demo.py                # Smoke test (không cần camera)
│   ├── requirements.txt
│   ├── data/
│   │   ├── raw_images/                 # Ảnh gốc
│   │   ├── processed_images/           # Contact sheet 4 bước
│   │   └── videos/                     # Video ngắn
│   ├── outputs/
│   │   ├── image_metadata.csv
│   │   └── image_event_log.csv
│   └── docs/
│       ├── PHAN_TICH_CODE_LAB6.md
│       ├── RUBRIC_LAB6.md
│       ├── CAU_HOI_HIEU_BAN_CHAT.md
│       ├── CHECKLIST_NOP_BAI.md
│       └── HUONG_DAN_CHAY_VA_QUAN_SAT.md
│
└── lab6_cv_as_iot_sensor_advanced/     # Lab 6 nâng cao (Parameter Experiment)
    ├── app.py                          # Backend FastAPI nâng cao
    ├── index.html                      # Dashboard nâng cao
    ├── run_lab6_advanced_demo.py       # Smoke test 11 bước
    ├── requirements.txt
    ├── data/
    │   ├── raw_images/
    │   ├── processed_images/           # Contact sheet 6 bước
    │   └── videos/
    ├── outputs/
    │   ├── image_metadata.csv
    │   ├── image_event_log.csv
    │   └── parameter_experiment_log.csv   # ← điểm cốt lõi của phần nâng cao
    └── docs/
        ├── HUONG_DAN_CHAY_VA_QUAN_SAT_NANG_CAO.md
        ├── CAU_HOI_VA_YEU_CAU_THAY_DOI_THAM_SO.md
        └── PHAN_TICH_CODE_NANG_CAO.md
```

---

## Lab 6 Cơ bản

### Mục tiêu

- Chạy camera stream từ laptop hoặc IP camera (có fallback stream mô phỏng)
- Chụp snapshot, ghi video ngắn, motion capture
- Tạo contact sheet 4 bước: **resize → grayscale → threshold → edge**
- Ghi `image_metadata.csv` và `image_event_log.csv`
- Quan sát toàn bộ pipeline trên dashboard

### Cài đặt và chạy

```bash
cd lab6_cv_as_iot_sensor

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Smoke test (không cần camera):**
```bash
python run_lab6_demo.py
```
Kiểm tra `RUN_TEST_LOG.txt` — dòng đầu phải là `LOCAL_PIPELINE_TEST_PASS`.

**Khởi động server:**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở trình duyệt: **http://127.0.0.1:8000/**

### API endpoints

| Endpoint | Mô tả |
|---|---|
| `GET /` | Dashboard HTML |
| `GET /video_feed?source=0` | MJPEG stream |
| `GET /snapshot?source=0` | Chụp 1 ảnh, chạy pipeline |
| `GET /record-video?seconds=5` | Ghi video ngắn |
| `GET /motion-capture?threshold=25&min_area=800` | Motion detection |
| `POST /upload-image` | Upload ảnh vào pipeline |
| `GET /metadata` | Đọc image_metadata.csv |
| `GET /events` | Đọc image_event_log.csv |
| `GET /latest` | Ảnh và event mới nhất |
| `GET /health` | Kiểm tra trạng thái server |
| `GET /docs` | Swagger UI |

### Camera source

| Giá trị | Ý nghĩa |
|---|---|
| `0` | Camera laptop (webcam mặc định) |
| `http://192.168.x.x:8080/video` | IP camera (IP Webcam app) |
| `rtsp://...` | RTSP stream |
| *(không có camera)* | Tự fallback sang stream mô phỏng |

---

## Lab 6 Nâng cao — Parameter Experiment

### Điểm khác biệt so với cơ bản

| Tính năng | Cơ bản | Nâng cao |
|---|---|---|
| Contact sheet | 4 bước | 6 bước (+Mask Combined, +Quality Info) |
| ROI | Không | Có — hiển thị khung xanh trên stream |
| Blur score | Không | `compute_blur_score()` — Laplacian variance |
| Quality events | LOW_LIGHT | LOW_LIGHT, OVER_EXPOSED, BLURRY, OK |
| `rule_used` trong event | Không | Có — truy vết điều kiện kích hoạt |
| Parameter log | Không | `parameter_experiment_log.csv` |
| Motion cooldown | Không | Có — `COOLDOWN_SKIP` event |
| **Motion notification** | Không | **Banner đỏ + toast popup + âm thanh** |

### Cài đặt và chạy

```bash
cd lab6_cv_as_iot_sensor_advanced

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
python run_lab6_advanced_demo.py
```

**Khởi động server (port 8001 để chạy song song với cơ bản):**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8001 --app-dir .
```

Mở trình duyệt: **http://127.0.0.1:8001/**

### Tham số thử nghiệm

| Tham số | Giá trị thử | Quan sát |
|---|---|---|
| `threshold_val` | 80, 120, 180 | Ảnh nhị phân sáng/tối |
| `canny_low/high` | 50/100, 80/160, 150/250 | Số biên và nhiễu |
| `roi` | full, `80,60,560,300` | brightness/blur_score thay đổi |
| `diff_threshold` | 15, 25, 40 | Độ nhạy motion |
| `min_area` | 500, 800, 1500 | Lọc motion nhỏ |
| `cooldown` | 0, 1, 5 (giây) | Alert fatigue |

### API endpoints bổ sung

| Endpoint | Mô tả |
|---|---|
| `GET /snapshot-advanced` | Snapshot với ROI + tham số đầy đủ |
| `POST /upload-image-advanced` | Upload ảnh vào advanced pipeline |
| `GET /motion-capture-advanced` | Motion với cooldown + ROI |
| `GET /parameter-experiments` | Đọc parameter_experiment_log.csv |

---

## Motion Notification System (Advanced)

Dashboard nâng cao có hệ thống thông báo realtime khi phát hiện chuyển động:

| Thành phần | Mô tả |
|---|---|
| **Sticky Banner** | Dải đỏ cố định đầu trang, hiển thị event type + motion score + timestamp. Có nút ✕ Đóng |
| **Toast Popup** | Popup góc phải dưới, tự động mờ sau 5 giây |
| **Âm thanh** | Beep ngắn qua Web Audio API (bật/tắt bằng checkbox) |
| **Polling 3s** | Kiểm tra event mới mỗi 3 giây — nhanh hơn dashboard refresh (6s) |

**Event → Thông báo:**
- `MOTION_DETECTED` → banner đỏ + toast đỏ + beep
- `COOLDOWN_SKIP` → banner xanh + toast xanh (event bị suppressed)
- `NO_SIGNIFICANT_MOTION` → toast xanh
- `LOW_LIGHT` / `BLURRY_IMAGE` / `OVER_EXPOSED_IMAGE` → toast đỏ

---

## Kết nối IP Camera qua điện thoại

1. Cài **IP Webcam** (Thyoni Tech / Pavel Khlebovich) từ Play Store
2. Mở app → nhấn **Start Server**
3. App hiện địa chỉ: `http://192.168.x.x:8080`
4. Test trên browser laptop: `http://192.168.x.x:8080/shot.jpg`
5. Nhập vào ô **Camera source**: `http://192.168.x.x:8080/video`

> Điện thoại và laptop phải cùng mạng WiFi.

---

## Event types

| Event | Điều kiện | Severity |
|---|---|---|
| `IMAGE_QUALITY_OK` | brightness 60–210 AND blur ≥ 80 | NORMAL |
| `LOW_LIGHT` | brightness < 60 | WARNING |
| `OVER_EXPOSED_IMAGE` | brightness > 210 | WARNING |
| `BLURRY_IMAGE` | blur_score < 80 | WARNING |
| `MOTION_DETECTED` | motion_score ≥ min_area | WARNING |
| `NO_SIGNIFICANT_MOTION` | motion_score < min_area | NORMAL |
| `COOLDOWN_SKIP` | cooldown còn active | NORMAL |
| `VIDEO_RECORDED` | ghi video xong | NORMAL |

---

## Yêu cầu hệ thống

```
Python >= 3.10
fastapi == 0.115.6
uvicorn[standard] == 0.34.0
opencv-python-headless == 4.10.0.84
numpy == 2.2.1
Pillow == 11.0.0
python-multipart == 0.0.20
```

---

## Vị trí trong chuỗi AIoT

| Lab | Vai trò | Sản phẩm kế thừa |
|---|---|---|
| Lab 5 | Đóng gói AI inference service | API, Docker, model |
| **Lab 6** | **Camera/ảnh như cảm biến IoT** | **Stream, metadata, event, dashboard** |
| Lab 7 | Object Detection | Class, confidence, bounding box |

Lab 6 **chưa nhận diện vật thể**. Lab 6 tập trung vào **đường đi của dữ liệu ảnh**: ảnh đến từ đâu, được lưu như thế nào, có metadata gì, sinh event gì. Ảnh đã qua Lab 6 với `IMAGE_QUALITY_OK` sẽ được đưa vào model detection ở Lab 7.
