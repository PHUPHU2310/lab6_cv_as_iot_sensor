"""
Smoke test for Lab 6 Advanced - Computer Vision as IoT Sensor: Parameter Experiment

Runs the full pipeline WITHOUT a web server or real camera.
After running, check:
  - RUN_TEST_LOG.txt        : status line + per-test results
  - data/raw_images/        : saved raw frames
  - data/processed_images/  : advanced 6-panel contact sheets
  - data/videos/            : short video clip
  - outputs/image_metadata.csv
  - outputs/image_event_log.csv
  - outputs/parameter_experiment_log.csv

Usage:
    python run_lab6_advanced_demo.py
"""

from pathlib import Path
import json
import sys

# Ensure we import from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import (
    simulated_frame,
    log_image_pipeline_advanced,
    motion_capture_advanced,
    record_short_video,
    METADATA_CSV, EVENT_CSV, PARAM_LOG_CSV,
    read_csv,
)

log_lines: list[str] = []
failures: list[str] = []


def record(label: str, result: dict) -> None:
    summary = {
        "test": label,
        "image_id": result.get("image_id", result.get("video_id", "?")),
        "event_type": (
            result.get("event", {}).get("event_type")
            or result.get("motion_event", {}).get("event_type")
            or "?"
        ),
    }
    log_lines.append(json.dumps(summary, ensure_ascii=False))
    print(f"  [{label}] {summary}")


def run_tests() -> None:
    # ------------------------------------------------------------------
    # Test 1: Full-frame snapshot with default parameters
    # ------------------------------------------------------------------
    print("\n--- Test 1: Snapshot default params (full frame, thresh=120, canny=80/160) ---")
    frame = simulated_frame(0)
    r = log_image_pipeline_advanced(
        frame, source_type="demo_default", device_id="simulated_cam",
        threshold_val=120, canny_low=80, canny_high=160,
        note="test1_default_params"
    )
    record("default_params", r)

    # ------------------------------------------------------------------
    # Test 2: Low threshold (threshold=80)
    # ------------------------------------------------------------------
    print("\n--- Test 2: Low threshold=80 ---")
    frame = simulated_frame(5)
    r = log_image_pipeline_advanced(
        frame, source_type="demo_thresh80", device_id="simulated_cam",
        threshold_val=80, canny_low=80, canny_high=160,
        note="test2_thresh80"
    )
    record("thresh_80", r)

    # ------------------------------------------------------------------
    # Test 3: High threshold (threshold=180)
    # ------------------------------------------------------------------
    print("\n--- Test 3: High threshold=180 ---")
    frame = simulated_frame(10)
    r = log_image_pipeline_advanced(
        frame, source_type="demo_thresh180", device_id="simulated_cam",
        threshold_val=180, canny_low=80, canny_high=160,
        note="test3_thresh180"
    )
    record("thresh_180", r)

    # ------------------------------------------------------------------
    # Test 4: Tight Canny (50/100)
    # ------------------------------------------------------------------
    print("\n--- Test 4: Canny edge 50/100 (low threshold → more edges) ---")
    frame = simulated_frame(15)
    r = log_image_pipeline_advanced(
        frame, source_type="demo_canny50", device_id="simulated_cam",
        threshold_val=120, canny_low=50, canny_high=100,
        note="test4_canny_50_100"
    )
    record("canny_50_100", r)

    # ------------------------------------------------------------------
    # Test 5: Wide Canny (150/250)
    # ------------------------------------------------------------------
    print("\n--- Test 5: Canny edge 150/250 (fewer edges) ---")
    frame = simulated_frame(20)
    r = log_image_pipeline_advanced(
        frame, source_type="demo_canny150", device_id="simulated_cam",
        threshold_val=120, canny_low=150, canny_high=250,
        note="test5_canny_150_250"
    )
    record("canny_150_250", r)

    # ------------------------------------------------------------------
    # Test 6: ROI — narrow region (upper-left quadrant only)
    # ------------------------------------------------------------------
    print("\n--- Test 6: ROI = (0,0,320,180) narrow region ---")
    frame = simulated_frame(25)
    r = log_image_pipeline_advanced(
        frame, source_type="demo_roi_narrow", device_id="simulated_cam",
        roi=(0, 0, 320, 180),
        threshold_val=120, canny_low=80, canny_high=160,
        note="test6_roi_0_0_320_180"
    )
    record("roi_narrow", r)

    # ------------------------------------------------------------------
    # Test 7: Motion capture — low diff_threshold (sensitive)
    # ------------------------------------------------------------------
    print("\n--- Test 7: Motion capture diff_threshold=15 (sensitive) ---")
    r = motion_capture_advanced(
        source="no_camera", seconds=2,
        diff_threshold=15, min_area=500, cooldown=0.0,
        threshold_val=120, canny_low=80, canny_high=160,
    )
    record("motion_dt15", r)

    # ------------------------------------------------------------------
    # Test 8: Motion capture — high diff_threshold (less sensitive)
    # ------------------------------------------------------------------
    print("\n--- Test 8: Motion capture diff_threshold=40 (less sensitive) ---")
    r = motion_capture_advanced(
        source="no_camera", seconds=2,
        diff_threshold=40, min_area=500, cooldown=0.0,
        threshold_val=120, canny_low=80, canny_high=160,
    )
    record("motion_dt40", r)

    # ------------------------------------------------------------------
    # Test 9: Motion capture — large min_area (only big motion)
    # ------------------------------------------------------------------
    print("\n--- Test 9: Motion capture min_area=1500 (only big motion) ---")
    r = motion_capture_advanced(
        source="no_camera", seconds=2,
        diff_threshold=25, min_area=1500, cooldown=0.0,
        threshold_val=120, canny_low=80, canny_high=160,
    )
    record("motion_ma1500", r)

    # ------------------------------------------------------------------
    # Test 10: Motion capture — cooldown=5s (most events suppressed)
    # ------------------------------------------------------------------
    print("\n--- Test 10: Motion capture cooldown=5s (event suppression) ---")
    r = motion_capture_advanced(
        source="no_camera", seconds=2,
        diff_threshold=15, min_area=500, cooldown=5.0,
        threshold_val=120, canny_low=80, canny_high=160,
    )
    record("motion_cooldown5", r)

    # ------------------------------------------------------------------
    # Test 11: Record short video
    # ------------------------------------------------------------------
    print("\n--- Test 11: Record short video (1s) ---")
    r = record_short_video("no_camera", seconds=1)
    record("video_1s", r)


def main() -> None:
    print("=" * 60)
    print("Lab 6 Advanced Smoke Test")
    print("=" * 60)

    try:
        run_tests()
        status = "LOCAL_PIPELINE_TEST_PASS"
    except Exception as exc:
        status = f"LOCAL_PIPELINE_TEST_FAIL: {exc}"
        failures.append(str(exc))
        print(f"\nFAILURE: {exc}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Verify CSV files exist and have data
    # ------------------------------------------------------------------
    print("\n--- CSV verification ---")
    for csv_path in [METADATA_CSV, EVENT_CSV, PARAM_LOG_CSV]:
        rows = read_csv(csv_path)
        label = csv_path.name
        print(f"  {label}: {len(rows)} rows")
        log_lines.append(json.dumps({"csv": label, "rows": len(rows)}, ensure_ascii=False))
        if len(rows) == 0:
            failures.append(f"{label} is empty after tests")

    if failures:
        status = f"LOCAL_PIPELINE_TEST_FAIL: {'; '.join(failures)}"

    # ------------------------------------------------------------------
    # Write log
    # ------------------------------------------------------------------
    log_content = status + "\n" + "\n".join(log_lines)
    Path("RUN_TEST_LOG.txt").write_text(log_content, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"STATUS: {status}")
    print("=" * 60)
    print("\nKiểm tra sau khi chạy:")
    print("  RUN_TEST_LOG.txt")
    print("  data/raw_images/         — ảnh gốc")
    print("  data/processed_images/   — contact sheet 6 bước")
    print("  data/videos/             — video ngắn")
    print("  outputs/image_metadata.csv")
    print("  outputs/image_event_log.csv")
    print("  outputs/parameter_experiment_log.csv")


if __name__ == "__main__":
    main()
