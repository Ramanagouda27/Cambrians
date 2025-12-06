from ultralytics import YOLO
import cv2
import numpy as np
import time
from collections import deque
from sklearn.cluster import KMeans

# ============================================================
# AUTO LANE DETECTION MODULE
# ============================================================

def auto_detect_lanes(frame, expected_lanes=3):
    """Automatically detect lane boundaries using edges + Hough + clustering."""
    h, w = frame.shape[:2]

    # Preprocessing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    edges = cv2.Canny(blur, 60, 150)

    # Mask bottom half
    mask = np.zeros_like(edges)
    mask[h//2:h, :] = 255
    edges = cv2.bitwise_and(edges, mask)

    # Hough vertical-ish lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=110,
                            minLineLength=80, maxLineGap=50)

    if lines is None:
        return [(i/expected_lanes, (i+1)/expected_lanes) for i in range(expected_lanes)]

    x_vals = []
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if abs(x1 - x2) < 45:   # vertical-ish
            x_vals.append([x1])
            x_vals.append([x2])

    # Not enough lines → fallback
    if len(x_vals) < expected_lanes + 1:
        return [(i/expected_lanes, (i+1)/expected_lanes) for i in range(expected_lanes)]

    # Cluster into N+1 boundaries
    kmeans = KMeans(n_clusters=expected_lanes+1, n_init="auto")
    kmeans.fit(x_vals)
    centers = sorted(kmeans.cluster_centers_.flatten())

    lane_bounds = []
    for i in range(expected_lanes):
        lane_bounds.append((centers[i] / w, centers[i+1] / w))

    return lane_bounds


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = r"C:\Users\Krishna B Dhamanekar\Downloads\3_lane_kelly_dashcam.mp4"
PROCESS_WIDTH = 960

CAUTION_DIST = 40
ALERT_DIST = 20
TTC_CRITICAL = 3

VEHICLE_IDS = {1,2,3,4,5,6,7}

distance_history = [deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)]
speed_history = [deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)]

lane_recommend_stable = None
recommend_timer = 0
STABLE_TIME = 2.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def estimate_speed(last_dist, curr_dist, last_time, curr_time):
    if last_dist is None or curr_dist is None:
        return 0.0

    dt = curr_time - last_time
    if dt <= 0:
        return 0.0

    speed = (last_dist - curr_dist) / dt   # positive = approaching

    if abs(speed) > 120:
        return 0.0

    return speed


def smooth(values):
    if len(values) == 0:
        return None
    return sum(values) / len(values)


# ============================================================
# LANE BLOCKS (BOTTOM RIGHT)
# ============================================================

def draw_lane_blocks(canvas, lane_count, best_lane):
    h, w = canvas.shape[:2]

    block_w = 55
    block_h = 120
    x0 = w - (lane_count * block_w) - 20
    y0 = h - block_h - 20

    for i in range(lane_count):
        x1 = x0 + i * block_w

        color = (120, 120, 120)   # grey by default
        if i == best_lane:
            color = (0, 255, 0)   # recommended

        cv2.rectangle(canvas, (x1, y0), (x1 + block_w, y0 + block_h),
                      color, 3)

        cv2.putText(canvas, f"L{i+1}",
                    (x1 + 10, y0 + block_h//2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)


# ============================================================
# MAIN ADAS SYSTEM
# ============================================================

def main():
    global lane_recommend_stable, recommend_timer

    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    try:
        model.to("cuda")
        print("Using GPU ✔")
    except:
        print("GPU not found → using CPU")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    num_lanes = 3  # expected number (auto-detection will refine)

    frame_count = 0
    last_time = time.time()
    last_lane_dist = [None] * num_lanes

    window = "ADAS Mirror View"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Press Q to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        h0, w0 = frame.shape[:2]
        scale = PROCESS_WIDTH / w0
        small = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * scale)))
        h, w = small.shape[:2]

        frame_count += 1

        # Auto detect lanes every 10 frames (stable)
        if frame_count % 10 == 0:
            LANE_BOUNDS = auto_detect_lanes(small, expected_lanes=num_lanes)
        # Safety fallback
        if frame_count == 1:
            LANE_BOUNDS = auto_detect_lanes(small, expected_lanes=num_lanes)

        curr_time = time.time()

        lane_dists = [None] * num_lanes
        lane_vehicles = [[] for _ in range(num_lanes)]

        # YOLO inference
        res = model(small, imgsz=PROCESS_WIDTH, conf=0.45, verbose=False)[0]

        if res.boxes:
            for box, score, cls in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls):
                if int(cls) not in VEHICLE_IDS:
                    continue

                x1,y1,x2,y2 = map(int, box.tolist())
                box_h = y2 - y1
                if box_h <= 0:
                    continue

                dist = (h * 6.0) / box_h  # pseudo-distance

                # Which lane?
                cx = (x1 + x2) / 2 / w
                lane = None
                for i,(lx,rx) in enumerate(LANE_BOUNDS):
                    if lx <= cx < rx:
                        lane = i
                        break
                if lane is None:
                    continue

                lane_vehicles[lane].append((x1,y1,x2,y2,dist))

                if lane_dists[lane] is None or dist < lane_dists[lane]:
                    lane_dists[lane] = dist

        # Smooth + speed + TTC
        lane_speed = [0]*num_lanes
        lane_ttc = [None]*num_lanes

        for i in range(num_lanes):

            if lane_dists[i] is not None:
                distance_history[i].append(lane_dists[i])

            smoothed = smooth(distance_history[i])
            lane_dists[i] = smoothed

            if smoothed is not None and last_lane_dist[i] is not None:
                sp = estimate_speed(last_lane_dist[i], smoothed, last_time, curr_time)
                speed_history[i].append(sp)
                lane_speed[i] = smooth(speed_history[i])

            if lane_speed[i] and lane_speed[i] > 0:
                lane_ttc[i] = smoothed / lane_speed[i]

        last_lane_dist = lane_dists.copy()
        last_time = curr_time

        # =====================================================
        # DRAWING OUTPUT
        # =====================================================

        out = small.copy()

        # Lane lines
        for lx,rx in LANE_BOUNDS:
            cv2.line(out, (int(lx*w), 0), (int(lx*w), h), (60,60,60), 2)
        cv2.line(out, (int(LANE_BOUNDS[-1][1]*w),0),
                 (int(LANE_BOUNDS[-1][1]*w),h), (60,60,60), 2)

        # Car boxes
        for i, veh_list in enumerate(lane_vehicles):
            for (x1,y1,x2,y2,dist) in veh_list:

                if dist <= ALERT_DIST:
                    color = (0,0,255)
                elif dist <= CAUTION_DIST:
                    color = (0,255,255)
                else:
                    color = (0,255,0)

                cv2.rectangle(out, (x1,y1),(x2,y2), color, 2)
                cv2.putText(out, f"{dist:.1f}m",
                            (x1, y1-8),
                            cv2.FONT_HERSHEY_DUPLEX, 0.7,
                            color, 2)

        # Top HUD
        hud_h = 70
        block_w = w//num_lanes

        for i in range(num_lanes):
            x0 = i * block_w
            x1 = x0 + block_w

            dist = lane_dists[i]

            if dist is None:
                color = (0,200,0)
                status = "CLEAR"
            elif dist <= ALERT_DIST:
                color = (0,0,255)
                status = f"ALERT {dist:.1f}m"
            elif dist <= CAUTION_DIST:
                color = (0,200,255)
                status = f"CAUTION {dist:.1f}m"
            else:
                color = (0,200,0)
                status = f"CLEAR {dist:.1f}m"

            cv2.rectangle(out, (x0,0),(x1,hud_h), color, -1)

            cv2.putText(out, f"Lane {i+1}", (x0+10, 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,255,255),2)

            cv2.putText(out, status, (x0+10, 56),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,255,255),2)

        # Lane Recommendation
        best_lane = None
        best_score = -9999
        for i in range(num_lanes):
            d = lane_dists[i] if lane_dists[i] is not None else 999999
            if d > best_score:
                best_lane = i
                best_score = d

        if lane_recommend_stable != best_lane:
            recommend_timer = curr_time
            lane_recommend_stable = best_lane
        else:
            if curr_time - recommend_timer < STABLE_TIME:
                best_lane = None

        # Lane blocks
        draw_lane_blocks(out, num_lanes, lane_recommend_stable)

        # Lower text
        cv2.putText(out, f"Lanes detected: {num_lanes}",
                    (20, h-65),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8,(255,255,0),2)

        if lane_recommend_stable is not None:
            cv2.putText(out, f"Recommended Lane: {lane_recommend_stable+1}",
                        (20, h-30),
                        cv2.FONT_HERSHEY_DUPLEX,1.0,(0,255,0),3)

        # Show
        cv2.imshow(window, out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
if __name__ == "__main__":
    main()