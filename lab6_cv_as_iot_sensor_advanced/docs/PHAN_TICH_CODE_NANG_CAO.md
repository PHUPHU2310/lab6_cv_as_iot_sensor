# Phân tích code nâng cao — Lab 6 Advanced

## 1. Tổng quan luồng dữ liệu

```
Camera / Upload
    │
    ▼
crop_roi()          ← xác định vùng phân tích (hoặc full frame)
    │
    ├──► compute_brightness()   → float  (mean grayscale)
    └──► compute_blur_score()   → float  (Laplacian variance)
    │
    ▼
create_processed_contact_sheet_advanced()
    ├── Panel 1: Original + ROI overlay
    ├── Panel 2: Grayscale
    ├── Panel 3: Threshold binary (threshold_val)
    ├── Panel 4: Canny edge (canny_low / canny_high)
    ├── Panel 5: Morphological mask (dilated threshold)
    └── Panel 6: Quality info text overlay
    │
    ▼
event_from_quality(brightness, blur_score)
    → LOW_LIGHT | OVER_EXPOSED_IMAGE | BLURRY_IMAGE | IMAGE_QUALITY_OK
    │
    ▼
append_csv(METADATA_CSV)          ← mô tả ảnh
append_csv(EVENT_CSV)             ← sự kiện chất lượng
append_csv(PARAM_LOG_CSV)         ← toàn bộ tham số + kết quả

(nếu là motion):
motion_capture_advanced()
    ├── frame difference → motion_score
    ├── cooldown check
    └── append_csv(EVENT_CSV) ← MOTION_DETECTED / NO_SIGNIFICANT_MOTION / COOLDOWN_SKIP
```

---

## 2. Phân tích từng hàm quan trọng

### `compute_brightness(frame_bgr, roi=None)`

- Chuyển frame (hoặc vùng ROI) sang grayscale.
- Trả về mean pixel value (0–255).
- **Ứng dụng**: phát hiện ảnh tối (< 60) hoặc quá sáng (> 210) trước khi đưa vào model.
- **Lưu ý**: ảnh tối có mean thấp nhưng có thể vẫn chứa thông tin ở vùng sáng cục bộ.

### `compute_blur_score(frame_bgr, roi=None)`

- Tính variance của Laplacian operator trên grayscale.
- Laplacian phát hiện cạnh; ảnh sắc nét → nhiều cạnh → variance cao.
- **Ứng dụng**: phát hiện camera rung, mất nét, kính bẩn.
- **Ngưỡng thực nghiệm**: < 80 → blurry; > 200 → sắc nét.
- **Câu hỏi**: tại sao ảnh chuyển động nhanh cũng có blur_score thấp?

### `crop_roi(frame_bgr, roi)`

- Nhận tuple `(x1, y1, x2, y2)` và trả về sub-array NumPy.
- Clamp tự động để tránh IndexError khi ROI nằm ngoài biên.
- **Ứng dụng**: chỉ phân tích vùng camera quan trọng (cửa vào, băng chuyền, lối đi).
- **Đánh đổi**: ROI nhỏ → ít nhiễu, nhanh hơn; nhưng bỏ sót sự kiện ngoài ROI.

### `create_processed_contact_sheet_advanced()`

- Tạo ảnh 3×2 (960×480 pixel) chứa 6 panel.
- **Panel 5 (Mask Combined)**: dùng morphological dilation để làm đầy lỗ hổng trong vùng sáng, sau đó overlay lên ảnh gốc (vùng tối bị làm mờ đi).
- **Panel 6 (Quality Info)**: text overlay màu xanh lá (NORMAL) hoặc đỏ (WARNING).
- **Tham số ảnh hưởng trực tiếp**: threshold_val, canny_low, canny_high.

### `event_from_quality(brightness, blur_score)`

- **Rule engine đơn giản** với thứ tự ưu tiên cố định:
  1. `brightness < 60` → `LOW_LIGHT`
  2. `brightness > 210` → `OVER_EXPOSED_IMAGE`
  3. `blur_score < 80` → `BLURRY_IMAGE`
  4. else → `IMAGE_QUALITY_OK`
- **`rule_used`** lưu điều kiện cụ thể đã kích hoạt → truy vết được tại sao event sinh ra.
- **Mở rộng**: để thêm `FLICKERING_IMAGE`, cần tính variance brightness giữa các frame liên tiếp.

### `motion_capture_advanced()`

- Thu thập frame trong `seconds` giây.
- Tính `motion_score` = tổng diện tích contour trong frame difference mask.
- **Tham số quan trọng**:
  - `diff_threshold`: pixel-level sensitivity (thấp = nhạy hơn)
  - `min_area`: ngưỡng diện tích tối thiểu để ghi nhận chuyển động
  - `cooldown`: thời gian tối thiểu giữa hai event (giảm alert fatigue)
- **`COOLDOWN_SKIP`**: event bị bỏ qua do cooldown còn active — được ghi log nhưng severity = NORMAL.
- **Lưu frame tốt nhất** (best_score cao nhất) thay vì tất cả frame → tiết kiệm lưu trữ.

### `log_image_pipeline_advanced()`

- Gọi cả 3 hàm append_csv: METADATA_CSV, EVENT_CSV, PARAM_LOG_CSV.
- **Điểm khác với basic**: ghi thêm `blur_score`, `roi`, `threshold_val`, `canny_low/high` vào metadata.
- **`parameter_experiment_log.csv`**: mỗi dòng = một lần thử nghiệm với đủ tham số và kết quả → có thể dùng để vẽ biểu đồ hoặc phân tích offline.

---

## 3. Sự khác biệt giữa Basic và Advanced

| Tiêu chí | Basic | Advanced |
|---|---|---|
| Contact sheet | 4 bước | 6 bước (có ROI overlay + quality info) |
| ROI | Không | Có; clamp tự động |
| Blur score | Không | Có (`compute_blur_score`) |
| Quality event | Chỉ LOW_LIGHT | LOW_LIGHT, OVER_EXPOSED, BLURRY, OK |
| rule_used | Không | Có trong event log |
| Parameter log | Không | `parameter_experiment_log.csv` |
| Motion cooldown | Không | Có; COOLDOWN_SKIP event |
| Stream ROI overlay | Không | Có (khung xanh lá trên stream) |

---

## 4. Câu hỏi kiểm tra hiểu code

| Hàm | Câu hỏi |
|---|---|
| `compute_brightness` | Nếu ảnh có vùng sáng nhỏ và phần còn lại rất tối, mean brightness có phản ánh vùng sáng đó không? |
| `compute_blur_score` | Camera rung nhanh làm Laplacian variance tăng hay giảm? |
| `crop_roi` | Nếu roi=(0,0,9999,9999), hàm trả về gì? |
| `event_from_quality` | Nếu brightness=55 và blur_score=50, event gì được sinh? Rule nào kích hoạt? |
| `motion_capture_advanced` | Nếu cooldown=10 và event vừa được sinh 3 giây trước, event tiếp theo có loại gì? |
| `create_processed_contact_sheet_advanced` | Thay đổi canny_low từ 80 xuống 20 làm thay đổi panel nào trong contact sheet? |

---

## 5. Kết nối sang Lab 7

- Ảnh có `event_type = IMAGE_QUALITY_OK` → sẵn sàng đưa vào object detection model.
- Ảnh có `LOW_LIGHT` hoặc `BLURRY_IMAGE` → cần preprocessing thêm hoặc bỏ qua.
- `parameter_experiment_log.csv` cho biết bộ tham số nào tạo ra ảnh chất lượng tốt nhất → dùng làm cấu hình mặc định cho Lab 7.
- ROI đã xác định trong Lab 6 → bounding box sơ bộ cho Lab 7 inference.
