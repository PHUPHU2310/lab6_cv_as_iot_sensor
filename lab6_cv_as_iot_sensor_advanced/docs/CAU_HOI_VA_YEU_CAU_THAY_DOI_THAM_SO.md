# Câu hỏi và yêu cầu thay đổi tham số — Lab 6 Nâng cao

## A. Câu hỏi phân tích tham số

### A1. Threshold ảnh xám

1. Khi threshold = 80, ảnh nhị phân trông như thế nào so với threshold = 180?
2. Giá trị nào giữ lại nhiều chi tiết hơn? Tại sao?
3. Nếu hệ thống dùng threshold quá cao để phát hiện vật thể sáng, loại sự kiện nào có thể bị bỏ sót?
4. Nên chọn threshold theo điều kiện ánh sáng hay cố định một giá trị? Giải thích.

### A2. Canny Edge

1. Khi Canny Low = 50, High = 100, ảnh biên có nhiều hơn hay ít hơn so với 150/250?
2. Cạnh dư thừa (nhiễu biên) ảnh hưởng đến độ tin cậy của visual event như thế nào?
3. Nếu camera đặt gần mặt đường ban đêm, bộ Canny nào phù hợp hơn?
4. Tại sao không nên dùng Low = 0? Điều gì xảy ra?

### A3. ROI — Region of Interest

1. Khi ROI hẹp, brightness và blur_score thay đổi như thế nào so với full frame?
2. Nếu ROI đặt vào vùng rất tối (bóng đổ), event gì sẽ được sinh?
3. ROI sai vị trí (đặt vào góc không có đối tượng) gây ra hậu quả gì đối với event downstream?
4. Trong hệ thống thực tế, ai nên quyết định ROI: kỹ sư hay hệ thống tự học?

### A4. Motion Detection

1. Diff threshold = 15 so với 40: tỷ lệ MOTION_DETECTED thay đổi thế nào?
2. Min_area = 500 so với 1500: loại chuyển động nào bị lọc bỏ?
3. Cooldown = 0 so với 5s: trong cùng một cảnh tĩnh, số event thay đổi thế nào?
4. Cooldown quá dài có thể gây bỏ sót sự kiện không? Cho ví dụ cụ thể.
5. Frame difference có phải object detection không? Giải thích sự khác biệt.

### A5. Chất lượng ảnh và quality event

1. Điều kiện nào tạo ra event `LOW_LIGHT`? `OVER_EXPOSED_IMAGE`? `BLURRY_IMAGE`?
2. Nếu ảnh vừa tối vừa mờ, rule nào được kích hoạt trước? Tại sao?
3. Nếu muốn thêm event `FLICKERING_IMAGE` (ảnh nhấp nháy do ánh đèn huỳnh quang), cần thêm logic gì vào `event_from_quality()`?
4. Blur score (Laplacian variance) có hoạt động tốt với ảnh chuyển động nhanh không?

---

## B. Yêu cầu thực hành bắt buộc

### B1. Bảng thử nghiệm threshold (nộp kèm báo cáo)

Thực hiện 3 snapshot với threshold = 80, 120, 180 (cùng cảnh, cùng ROI, cùng Canny).

| Threshold | Mô tả vùng trắng trong panel Threshold | Event sinh ra | Nhận xét |
|---|---|---|---|
| 80 | | | |
| 120 | | | |
| 180 | | | |

### B2. Bảng thử nghiệm Canny edge (nộp kèm báo cáo)

| Canny Low/High | Số biên ước lượng (nhiều/trung bình/ít) | Mức nhiễu | Nhận xét |
|---|---|---|---|
| 50/100 | | | |
| 80/160 | | | |
| 150/250 | | | |

### B3. Bảng so sánh motion cấu hình (nộp kèm báo cáo)

| Diff Threshold | Min Area | Cooldown | Event type | Motion score | Nhận xét |
|---|---|---|---|---|---|
| 15 | 500 | 0 | | | |
| 25 | 800 | 1s | | | |
| 40 | 1500 | 5s | | | |

### B4. Bảng ROI (nộp kèm báo cáo)

| ROI | Brightness | Blur Score | Event | Nhận xét |
|---|---|---|---|---|
| Full frame | | | | |
| ROI hẹp (vùng chính) | | | | |
| ROI sai vị trí | | | | |

---

## C. Câu hỏi hiểu bản chất (trả lời ngắn trong báo cáo)

1. Vì sao cần ghi `rule_used` trong event log, không chỉ ghi event_type?
2. Trong hệ thống AIoT thực tế, ai kiểm tra `parameter_experiment_log.csv` và dùng để làm gì?
3. Nếu camera ngoài trời, threshold ban ngày và ban đêm có nên khác nhau không?
4. Nếu blur_score giảm đột ngột trong 5 phút, điều đó có thể có nghĩa gì?
5. Sang Lab 7, ảnh có event `LOW_LIGHT` hoặc `BLURRY_IMAGE` sẽ ảnh hưởng đến object detection như thế nào?
6. Nếu motion event quá nhiều (alert fatigue), nên điều chỉnh diff_threshold, min_area hay cooldown trước? Vì sao?
7. Tại sao không nên lưu tất cả frame vào raw_images mà chỉ lưu frame tốt nhất từ motion capture?
8. Nếu chỉ xử lý ROI, hệ thống có thể bỏ sót sự kiện ở vùng ngoài ROI không? Đánh đổi gì?
