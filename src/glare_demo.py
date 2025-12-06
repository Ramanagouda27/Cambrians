import cv2
import numpy as np

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
# 0 = webcam, or put path to a night / high-beam video
VIDEO_PATH = 0

WINDOW_NAME = "Glare Reduction Demo"

# Detection thresholds
GLARE_V_MIN = 230         # brightness threshold  (0–255) – raise if it selects too much
GLARE_S_MAX = 70          # saturation upper bound (white-ish)
GLARE_MIN_AREA = 80       # ignore tiny spots

# How much to reduce brightness in strong glare regions
MAX_DARKEN_FACTOR = 0.5   # 0.5 => keep 50% brightness at the very center

# ---------------------------------------------------
# GLARE DETECTION (NO ROI – FULL FRAME)
# ---------------------------------------------------

def detect_glare_mask(frame):
    """
    Detect bright white glare spots (e.g., flash / high beam) in the frame.
    Returns a binary mask (uint8, 0 or 255) over the full image.
    """
    h, w = frame.shape[:2]

    # Work in HSV for brightness / color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # very bright
    bright_mask = cv2.inRange(v_ch, GLARE_V_MIN, 255)
    # low saturation (nearly white)
    low_sat_mask = cv2.inRange(s_ch, 0, GLARE_S_MAX)

    # combine
    mask = cv2.bitwise_and(bright_mask, low_sat_mask)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove tiny blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= GLARE_MIN_AREA:
            clean_mask[labels == i] = 255

    return clean_mask


def apply_glare_reduction(frame, glare_mask):
    """
    Darken only the glare regions smoothly, preserving texture.
    """
    # Blur mask edges → soft transition
    mask_blur = cv2.GaussianBlur(glare_mask, (31, 31), 0).astype(np.float32) / 255.0
    mask_blur = cv2.merge([mask_blur, mask_blur, mask_blur])  # 3 channels

    # alpha: 1 outside glare, (1 - MAX_DARKEN_FACTOR) inside full glare
    # e.g., MAX_DARKEN_FACTOR=0.5 → alpha goes from 1.0 to 0.5
    alpha = 1.0 - MAX_DARKEN_FACTOR * mask_blur

    frame_f = frame.astype(np.float32)
    out = (frame_f * alpha).astype(np.uint8)
    return out

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open camera / video.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    prev_mask = None
    print("Glare demo v2 running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional resize for speed
        h0, w0 = frame.shape[:2]
        scale = 960.0 / float(w0)
        frame = cv2.resize(frame, (960, int(h0 * scale)))
        h, w = frame.shape[:2]

        # --- Detect glare over FULL frame ---
        raw_mask = detect_glare_mask(frame)

        # Temporal smoothing (reduce flicker)
        if prev_mask is None:
            smooth_mask = raw_mask
        else:
            smooth_mask = cv2.addWeighted(prev_mask, 0.6, raw_mask, 0.4, 0)
        prev_mask = smooth_mask

        # --- Apply reduction ---
        cleaned = apply_glare_reduction(frame, smooth_mask)

        # Side-by-side compare
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
        combined[:, :w] = frame
        combined[:, w:] = cleaned

        # Titles
        cv2.putText(combined, "Original", (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(combined, "De-glared View", (w + 20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)

        # Optional: draw a thin outline around detected glare (for explanation)
        contours, _ = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < GLARE_MIN_AREA:
                continue
            cv2.drawContours(combined[:, :w], [c], -1, (0, 0, 255), 1)
            cv2.drawContours(combined[:, w:], [c], -1, (0, 0, 255), 1)

        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()