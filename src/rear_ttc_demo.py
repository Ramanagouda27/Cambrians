from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

# ============================================================
# CONFIG
# ============================================================

# Set this to your rear / side traffic video
# Example:
# VIDEO_PATH = r"C:\Users\Krishna B Dhamanekar\Downloads\rear_view_test.mp4"
VIDEO_PATH = r"C:\Users\Krishna B Dhamanekar\Downloads\rear_video.mp4"

# If you want webcam instead, set:
# VIDEO_PATH = 0

MODEL_PATH = "yolov8n.pt"      # you already have this
PROCESS_WIDTH = 960            # resize width for speed

# COCO IDs for vehicles
VEHICLE_IDS = {1, 2, 3, 4, 5, 6, 7}  # bicycle, car, motorbike, airplane, bus, train, truck

# Distance & TTC thresholds (tune these visually)
CAUTION_TTC = 5.0   # seconds  -> yellow
ALERT_TTC   = 3.0   # seconds  -> red

# Smoothing
HISTORY_LEN = 7   # how many frames to smooth distance & TTC

# ============================================================
# HELPERS
# ============================================================

def estimate_distance(box_h, frame_h, k=6.0):
    """
    Pseudo-distance in meters.
    Larger bbox height -> closer -> smaller distance.
    k is a tuning constant; increase to make all distances bigger.
    """
    if box_h <= 0:
        return None
    return (frame_h * k) / float(box_h)


def estimate_speed(last_d, curr_d, last_t, curr_t):
    """
    Relative speed in m/s (positive = approaching).
    """
    if last_d is None or curr_d is None:
        return 0.0

    dt = curr_t - last_t
    if dt <= 0:
        return 0.0

    v = (last_d - curr_d) / dt

    # clamp crazy values
    if abs(v) > 120:
        return 0.0

    return v


def smooth(history_deque):
    if len(history_deque) == 0:
        return None
    return sum(history_deque) / len(history_deque)


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    try:
        model.to("cuda")
        print("Using GPU ✔")
    except Exception:
        print("GPU not found -> using CPU")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open video / camera.")
        return

    window = "Rear Time-to-Collision Demo"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # history buffers
    dist_hist = deque(maxlen=HISTORY_LEN)
    ttc_hist  = deque(maxlen=HISTORY_LEN)

    last_time = time.time()
    last_dist = None

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------------------
        # Resize frame for speed
        # ---------------------------------------
        h0, w0 = frame.shape[:2]
        scale = PROCESS_WIDTH / float(w0)
        frame_small = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * scale)))
        h, w = frame_small.shape[:2]

        curr_time = time.time()

        # ---------------------------------------
        # Detect vehicles
        # ---------------------------------------
        res = model(frame_small, imgsz=PROCESS_WIDTH, conf=0.45, iou=0.45, verbose=False)[0]

        closest_dist = None
        closest_box = None

        if res.boxes is not None:
            for box, score, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                cls_id = int(cls)
                if cls_id not in VEHICLE_IDS:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                box_h = y2 - y1
                box_w = x2 - x1
                if box_h <= 0 or box_w <= 0:
                    continue

                # We only care about vehicles roughly in the lower half
                # (i.e., behind us, on the road).
                cy = (y1 + y2) / 2.0
                if cy < h * 0.35:   # ignore objects high up in frame
                    continue

                # Estimate distance
                d = estimate_distance(box_h, h, k=6.0)
                if d is None:
                    continue

                if (closest_dist is None) or (d < closest_dist):
                    closest_dist = d
                    closest_box = (x1, y1, x2, y2)

        # ---------------------------------------
        # TTC & speed estimation
        # ---------------------------------------
        if closest_dist is not None:
            dist_hist.append(closest_dist)
            smoothed_dist = smooth(dist_hist)
        else:
            smoothed_dist = smooth(dist_hist)

        if smoothed_dist is not None and last_dist is not None:
            rel_speed = estimate_speed(last_dist, smoothed_dist, last_time, curr_time)
        else:
            rel_speed = 0.0

        if rel_speed > 0:
            ttc = smoothed_dist / rel_speed
        else:
            ttc = None

        if ttc is not None and 0 < ttc < 60:  # ignore insane values
            ttc_hist.append(ttc)
            smoothed_ttc = smooth(ttc_hist)
        else:
            smoothed_ttc = smooth(ttc_hist)

        last_dist = smoothed_dist
        last_time = curr_time

        # ---------------------------------------
        # Draw HUD
        # ---------------------------------------
        out = frame_small.copy()

        # Draw closest vehicle
        if closest_box is not None and smoothed_dist is not None:
            x1, y1, x2, y2 = closest_box

            # color by TTC
            if smoothed_ttc is not None and smoothed_ttc <= ALERT_TTC:
                color = (0, 0, 255)      # red
            elif smoothed_ttc is not None and smoothed_ttc <= CAUTION_TTC:
                color = (0, 255, 255)    # yellow
            else:
                color = (0, 255, 0)      # green

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            cv2.putText(out, f"{smoothed_dist:5.1f} m",
                        (x1, max(y1 - 8, 25)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)

        # Right-side HUD panel
        hud_w = 260
        hud_x0 = w - hud_w
        cv2.rectangle(out, (hud_x0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(out[:, hud_x0:], 0.3, np.zeros_like(out[:, hud_x0:]), 0.7, 0, out[:, hud_x0:])

        cv2.putText(out, "Rear Safety HUD", (hud_x0 + 20, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

        # Distance info
        if smoothed_dist is not None:
            cv2.putText(out, f"Dist: {smoothed_dist:5.1f} m",
                        (hud_x0 + 20, 90),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(out, "Dist: -- m",
                        (hud_x0 + 20, 90),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # Relative speed
        cv2.putText(out, f"Rel v: {rel_speed:5.1f} m/s",
                    (hud_x0 + 20, 130),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        # TTC + status
        if smoothed_ttc is None:
            status_text = "SAFE"
            status_color = (0, 255, 0)
            ttc_text = "TTC: -- s"
        else:
            ttc_text = f"TTC: {smoothed_ttc:4.1f} s"
            if smoothed_ttc <= ALERT_TTC:
                status_text = "ALERT"
                status_color = (0, 0, 255)
            elif smoothed_ttc <= CAUTION_TTC:
                status_text = "CAUTION"
                status_color = (0, 255, 255)
            else:
                status_text = "SAFE"
                status_color = (0, 255, 0)

        cv2.putText(out, ttc_text,
                    (hud_x0 + 20, 170),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
        cv2.putText(out, status_text,
                    (hud_x0 + 20, 210),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 3)

        # Small note
        cv2.putText(out, "Press 'q' to quit",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(window, out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
if __name__ == "__main__":
    main()