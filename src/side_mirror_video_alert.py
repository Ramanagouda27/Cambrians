from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque

# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------

# Use your side-mirror video here
VIDEO_PATH = r"C:\Users\Krishna B Dhamanekar\Downloads\overtaking_alert.mp4"

# 960 width is enough for speed + quality
PROCESS_WIDTH = 960

# Model path:
# - for now: "yolov8s.pt" (more accurate than n)
# - later:  r"runs/detect/mirror_train/weights/best.pt" after you train
MODEL_PATH = "yolov8s.pt"

# Which YOLO classes count as vehicles (COCO IDs)
VEHICLE_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorbike, bus, truck

# Mirror ROI (relative, inside frame) – tune if needed
# (x1, y1, x2, y2) in fractions of width/height
MIRROR_ROI = (0.20, 0.25, 0.80, 0.95)

# Distance + TTC thresholds
CAUTION_DIST = 35.0      # m
ALERT_DIST   = 20.0      # m
BLIND_SPOT_MIN = 8.0     # m  (too close to see in mirror)
BLIND_SPOT_MAX = 22.0    # m

# value in seconds for "fast overtake" warning
TTC_ALERT = 3.0

# smoothing windows
DIST_HISTORY = deque(maxlen=6)
REL_V_HISTORY = deque(maxlen=6)


# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

def estimate_distance(box_h, frame_h):
    """
    Very simple pseudo-distance:
    larger box -> closer.
    You can tune SCALE factor after seeing your video.
    """
    if box_h <= 0:
        return None
    SCALE = 6.5  # tune this per video
    dist = (frame_h * SCALE) / float(box_h)
    return dist


def estimate_relative_speed(dists, times):
    """Return smoothed relative speed (m/s). Positive = approaching."""
    if len(dists) < 2:
        return 0.0

    d1, d2 = dists[-2], dists[-1]
    t1, t2 = times[-2], times[-1]
    dt = t2 - t1
    if dt <= 0:
        return 0.0

    v = (d1 - d2) / dt  # approach speed
    # Clamp silly spikes
    if abs(v) > 100:
        return 0.0
    return v


def draw_text(img, text, org, color, scale=0.8, thickness=2):
    cv2.putText(img, text, org,
                cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness,
                lineType=cv2.LINE_AA)


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():
    print("Loading YOLO model:", MODEL_PATH)
    model = YOLO(MODEL_PATH)

    try:
        model.to("cuda")
        print("Using GPU ✔")
    except Exception:
        print("GPU not available → using CPU")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open video:", VIDEO_PATH)
        return

    window = "Side Mirror Blind-spot & Overtake Alert"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # for speed / TTC history
    time_history = deque(maxlen=6)

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]
        scale = PROCESS_WIDTH / float(w0)
        frame = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * scale)))
        h, w = frame.shape[:2]

        # Mirror ROI (white box)
        mx1 = int(MIRROR_ROI[0] * w)
        my1 = int(MIRROR_ROI[1] * h)
        mx2 = int(MIRROR_ROI[2] * w)
        my2 = int(MIRROR_ROI[3] * h)

        mirror_view = frame[my1:my2, mx1:mx2].copy()
        mh, mw = mirror_view.shape[:2]

        now = time.time()
        time_history.append(now)

        # ----------------------------------------
        # YOLO inference only on the mirror region
        # ----------------------------------------
        res = model(mirror_view, imgsz=640, conf=0.5, iou=0.45, verbose=False)[0]

        nearest_dist = None
        nearest_box = None

        if res.boxes is not None:
            for box, score, cls in zip(res.boxes.xyxy,
                                       res.boxes.conf,
                                       res.boxes.cls):

                if int(cls) not in VEHICLE_IDS:
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                bw = x2 - x1
                bh = y2 - y1

                # ignore tiny blobs
                if bw * bh < 400:
                    continue

                dist = estimate_distance(bh, mh)
                if dist is None:
                    continue

                # keep nearest one for main decision
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_box = (x1, y1, x2, y2)

        # ------------------------------
        # update distance history
        # ------------------------------
        if nearest_dist is not None:
            DIST_HISTORY.append(nearest_dist)
        else:
            # decay history slowly so there is no sudden 0
            if DIST_HISTORY:
                DIST_HISTORY.append(DIST_HISTORY[-1] + 1.5)  # drifting away

        # smooth distance
        if DIST_HISTORY:
            smooth_dist = sum(DIST_HISTORY) / len(DIST_HISTORY)
        else:
            smooth_dist = None

        # relative speed
        rel_v = estimate_relative_speed(DIST_HISTORY, time_history)
        REL_V_HISTORY.append(rel_v)
        if REL_V_HISTORY:
            rel_v_smooth = sum(REL_V_HISTORY) / len(REL_V_HISTORY)
        else:
            rel_v_smooth = 0.0

        # TTC
        if smooth_dist is not None and rel_v_smooth > 0.1:
            ttc = smooth_dist / rel_v_smooth
        else:
            ttc = None

        # ------------------------------
        # Status logic
        # ------------------------------
        status_text = ""
        status_color = (0, 255, 0)  # default green

        if nearest_dist is None:
            status_text = "CLEAR: NO VEHICLE"
            status_color = (0, 255, 0)
        else:
            # Blind spot detection (close but not super close)
            if BLIND_SPOT_MIN <= smooth_dist <= BLIND_SPOT_MAX and abs(rel_v_smooth) < 0.5:
                status_text = f"VEHICLE IN BLIND SPOT ({smooth_dist:.1f}m)"
                status_color = (0, 255, 255)  # yellow
            # Fast overtake
            elif ttc is not None and ttc < TTC_ALERT:
                status_text = f"ALERT: FAST OVERTAKE! TTC {ttc:.1f}s"
                status_color = (0, 0, 255)  # red
            # Normal caution if just close
            elif smooth_dist <= ALERT_DIST:
                status_text = f"ALERT: VEHICLE CLOSE ({smooth_dist:.1f}m)"
                status_color = (0, 0, 255)
            elif smooth_dist <= CAUTION_DIST:
                status_text = f"CAUTION: VEHICLE AT {smooth_dist:.1f}m"
                status_color = (0, 255, 255)
            else:
                status_text = f"CLEAR: VEHICLE AT {smooth_dist:.1f}m"
                status_color = (0, 255, 0)

        # ------------------------------
        # DRAWING
        # ------------------------------
        vis = frame.copy()

        # mirror ROI box
        cv2.rectangle(vis, (mx1, my1), (mx2, my2), (220, 220, 220), 2)
        draw_text(vis, "Mirror Area", (mx1 + 5, my1 + 25), (255, 255, 255), 0.6, 2)

        # draw nearest vehicle box back in full frame
        if nearest_box is not None:
            x1, y1, x2, y2 = nearest_box
            # map back to full coordinates
            gx1, gy1 = mx1 + x1, my1 + y1
            gx2, gy2 = mx1 + x2, my1 + y2

            box_color = (0, 255, 0)
            if smooth_dist is not None:
                if smooth_dist <= ALERT_DIST:
                    box_color = (0, 0, 255)
                elif smooth_dist <= CAUTION_DIST:
                    box_color = (0, 255, 255)

            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), box_color, 2)
            if smooth_dist is not None:
                draw_text(vis, f"{smooth_dist:.1f}m",
                          (gx1, max(gy1 - 8, 20)),
                          box_color, 0.7, 2)

        # Top-right HUD title + status
        title = "Side Mirror Blind-spot & Overtake Alert"
        draw_text(vis, title, (20, 40), (255, 255, 0), 0.9, 2)

        draw_text(vis, status_text, (20, 75), status_color, 0.9, 2)

        # bottom-left: relative speed (m/s and km/h)
        rel_kmh = rel_v_smooth * 3.6
        draw_text(vis, f"Rel v: {rel_v_smooth:5.1f} m/s (~{rel_kmh:4.1f} km/h)",
                  (20, h - 30), (255, 255, 255), 0.7, 2)

        cv2.imshow(window, vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()