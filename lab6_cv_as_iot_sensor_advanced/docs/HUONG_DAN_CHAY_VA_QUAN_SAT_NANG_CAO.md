# Hướng dẫn chạy và quan sát — Lab 6 Nâng cao

## 1. Cài đặt môi trường

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Kiểm tra: không có lỗi import `fastapi`, `cv2`, `PIL`.

---

## 2. Chạy smoke test (không cần camera)

```bash
python run_lab6_advanced_demo.py
```

Kết quả cần quan sát:

| File / Thư mục | Nội dung kỳ vọng |
|---|---|
| `RUN_TEST_LOG.txt` | Dòng đầu: `LOCAL_PIPELINE_TEST_PASS` |
| `data/raw_images/` | Ít nhất 11 ảnh `.jpg` |
| `data/processed_images/` | Ít nhất 11 file `*_processed_advanced.jpg` |
| `data/videos/` | Ít nhất 1 file `.mp4` |
| `outputs/image_metadata.csv` | Có header + dữ liệu |
| `outputs/image_event_log.csv` | Có header + dữ liệu |
| `outputs/parameter_experiment_log.csv` | Có header + 11 dòng thử nghiệm |

---

## 3. Khởi động web dashboard

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Mở trình duyệt: **http://127.0.0.1:8000/**

---

## 4. Hướng dẫn thao tác trên dashboard

### 4.1 Bật stream

- Để **Source** = `0` cho camera laptop.
- Nhập URL nếu dùng IP camera: `http://192.168.x.x:8080/video`
- Để trống ROI = toàn khung; nhập `x1,y1,x2,y2` để chọn vùng.
- Nhấn **▶ Bật stream** → khung xanh lá hiển thị ROI trên stream.

### 4.2 Chụp snapshot với tham số khác nhau

Thử lần lượt các bộ tham số:

| Bộ | Threshold | Canny Low | Canny High | Cần quan sát |
|---|---|---|---|---|
| A | 80 | 80 | 160 | Nhiều vùng sáng trong threshold panel |
| B | 120 | 80 | 160 | Tham số mặc định, cân bằng |
| C | 180 | 80 | 160 | Chỉ vùng rất sáng còn lại |
| D | 120 | 50 | 100 | Nhiều biên hơn ở Canny panel |
| E | 120 | 150 | 250 | Ít biên hơn, chỉ biên rõ |

### 4.3 Chọn ROI hẹp

1. Quan sát khung ảnh stream để xác định vùng quan trọng.
2. Nhập ROI: ví dụ `100,50,540,310`.
3. Nhấn **📷 Chụp snapshot**.
4. Quan sát ảnh panel 1 (khung xanh lá) và panel 6 (brightness, blur_score chỉ tính trong ROI).

### 4.4 Motion capture

Vào section **Motion Capture**, thử 3 cấu hình:

| Cấu hình | Diff Threshold | Min Area | Cooldown | Kỳ vọng |
|---|---|---|---|---|
| Nhạy | 15 | 500 | 0 | Nhiều MOTION_DETECTED |
| Cân bằng | 25 | 800 | 1s | Một số event |
| Thô | 40 | 1500 | 5s | Ít event hơn |

### 4.5 Upload ảnh

- Chọn ảnh từ máy, chỉnh tham số, nhấn **⬆ Upload & xử lý ảnh**.
- Quan sát contact sheet 6 bước và quality event.

---

## 5. Đọc parameter_experiment_log.csv

Mở file `outputs/parameter_experiment_log.csv` và so sánh:

- Cùng ảnh, threshold khác nhau → `brightness` không đổi, nhưng contact sheet thay đổi.
- ROI hẹp → brightness và blur_score khác full frame.
- Cooldown cao → event_type chuyển sang `COOLDOWN_SKIP`.
- `rule_used` cho biết rule nào đã kích hoạt event.
