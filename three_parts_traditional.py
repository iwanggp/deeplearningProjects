"""
Traditional 3-part washer inspection using OpenCV + NumPy only.

Pipeline (fixed camera):
1) Crop a fixed ROI around the washer stack.
2) Use edge projection along Y to split into stacked segments.
3) Classify each segment by red color ratio + edge density.
4) Report missing/extra and draw results.
"""

import cv2
import numpy as np


CFG = {
    # Default ROI if not using interactive selection.
    # Format: (x, y, w, h)
    "roi": (120, 60, 360, 260),
    # Edge detection
    "canny_t1": 40,
    "canny_t2": 120,
    # 1D smoothing for projection
    "proj_smooth_ksize": 9,
    "proj_thresh_percentile": 70,
    "min_segment_height": 8,
    "merge_gap": 4,
    # Red fiber thresholds (HSV)
    "red_hsv_ranges": [
        ((0, 80, 50), (10, 255, 255)),
        ((170, 80, 50), (180, 255, 255)),
    ],
    "red_ratio_thresh": 0.18,
    # Spring washer tends to have denser edges
    "spring_edge_density_thresh": 0.12,
}


def _smooth_1d(signal, ksize):
    if ksize <= 1:
        return signal
    kernel = np.ones(ksize, dtype=np.float32) / ksize
    return np.convolve(signal, kernel, mode="same")


def _find_segments(proj, thresh, min_height, merge_gap):
    segments = []
    in_seg = False
    start = 0
    for i, val in enumerate(proj):
        if val >= thresh and not in_seg:
            in_seg = True
            start = i
        elif val < thresh and in_seg:
            end = i - 1
            if end - start + 1 >= min_height:
                segments.append((start, end))
            in_seg = False
    if in_seg:
        end = len(proj) - 1
        if end - start + 1 >= min_height:
            segments.append((start, end))

    if not segments:
        return []

    merged = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if seg[0] - prev[1] <= merge_gap:
            merged[-1] = (prev[0], seg[1])
        else:
            merged.append(seg)
    return merged


def _red_mask(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lo, hi) in ranges:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def _classify_segment(seg_img, seg_edges, seg_hsv):
    h, w = seg_img.shape[:2]
    area = max(h * w, 1)

    red = _red_mask(seg_hsv, CFG["red_hsv_ranges"])
    red_ratio = float(np.count_nonzero(red)) / area

    edge_density = float(np.count_nonzero(seg_edges)) / area

    if red_ratio >= CFG["red_ratio_thresh"]:
        return "red_fiber", red_ratio, edge_density
    if edge_density >= CFG["spring_edge_density_thresh"]:
        return "spring_washer", red_ratio, edge_density
    return "flat_washer", red_ratio, edge_density


def detect(image_path, roi=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if roi is None:
        roi = CFG["roi"]
    x, y, w, h = roi
    roi = img[y : y + h, x : x + w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, CFG["canny_t1"], CFG["canny_t2"])

    proj = np.sum(edges > 0, axis=1).astype(np.float32)
    proj = _smooth_1d(proj, CFG["proj_smooth_ksize"])
    proj_thresh = np.percentile(proj, CFG["proj_thresh_percentile"])
    proj_thresh = max(proj_thresh, 2.0)

    segments = _find_segments(
        proj, proj_thresh, CFG["min_segment_height"], CFG["merge_gap"]
    )

    results = []
    for (y1, y2) in segments:
        seg_img = roi[y1 : y2 + 1, :]
        seg_edges = edges[y1 : y2 + 1, :]
        seg_hsv = hsv[y1 : y2 + 1, :]
        label, red_ratio, edge_density = _classify_segment(
            seg_img, seg_edges, seg_hsv
        )
        results.append(
            {
                "label": label,
                "bbox": (x, y + y1, w, y2 - y1 + 1),
                "red_ratio": red_ratio,
                "edge_density": edge_density,
            }
        )

    return img, results, segments


def evaluate(results):
    expected = ["flat_washer", "spring_washer", "red_fiber"]
    counts = {k: 0 for k in expected}
    for r in results:
        counts[r["label"]] += 1

    missing = [k for k in expected if counts[k] == 0]
    extra = [k for k in expected if counts[k] > 1]

    ok = (len(results) == 3) and not missing and not extra
    return ok, missing, extra


def draw_results(img, results, ok, missing, extra):
    out = img.copy()
    colors = {
        "flat_washer": (0, 255, 0),
        "spring_washer": (0, 255, 255),
        "red_fiber": (255, 0, 255),
    }

    for r in results:
        x, y, w, h = r["bbox"]
        color = colors.get(r["label"], (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f"{r['label']} rr={r['red_ratio']:.2f} ed={r['edge_density']:.2f}"
        cv2.putText(out, text, (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    status = "OK" if ok else "ERROR"
    msg = status
    if missing:
        msg += f" missing={','.join(missing)}"
    if extra:
        msg += f" extra={','.join(extra)}"
    cv2.putText(out, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if ok else (0, 0, 255), 2)
    return out


def _select_roi(image):
    # OpenCV built-in ROI selector
    r = cv2.selectROI("Select ROI", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    return tuple(int(v) for v in r)


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python three_parts_traditional.py <image_path> [--roi]")
        return

    image_path = sys.argv[1]
    use_roi_picker = "--roi" in sys.argv[2:]

    roi = None
    if use_roi_picker:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Cannot read image: {image_path}")
            return
        roi = _select_roi(img)
        if roi[2] == 0 or roi[3] == 0:
            print("Empty ROI selected.")
            return
        print(f"Selected ROI: {roi}")

    img, results, segments = detect(image_path, roi=roi)
    ok, missing, extra = evaluate(results)
    out = draw_results(img, results, ok, missing, extra)
    cv2.imwrite("result_traditional.png", out)
    print("Saved: result_traditional.png")
    print(f"Segments: {len(segments)} -> {[r['label'] for r in results]}")
    print(f"Status: {'OK' if ok else 'ERROR'} missing={missing} extra={extra}")


if __name__ == "__main__":
    main()
