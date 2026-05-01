"""
KIDS AIR DRAWING GAME — PREMIUM FINAL V6
Stable Gesture + Modern Popup Scorecard
"""

import math, os, urllib.request
from dataclasses import dataclass, field
import cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

from adaptive_profile import AdaptiveProfile
from adaptive_trainer import generate_trainer_state
from progress_analytics import session_summary


class Config:
    MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)),"hand_landmarker.task")

    OPEN_HOLD=12
    DRAW_START_HOLD=4
    DRAW_END_HOLD=8
    MIN_STROKE_PTS=20

    GUIDE_RADIUS=95
    GUIDE_COLOR=(70,210,140)
    GUIDE_DIM=(40,80,60)

    INK_COLOR=(0,210,255)
    INK_THICKNESS=7

    DTW_DECAY_K=1.5
    W_ACCURACY=0.55
    W_COMPLETION=0.25
    W_SMOOTHNESS=0.20

    STAR_5=88
    STAR_4=72
    STAR_3=52
    STAR_2=32

    RESULT_FRAMES=110


INDEX_TIP=8
INDEX_PIP=6
INDEX_MCP=5
PINKY_MCP=17

HAND_CONNECTIONS=[
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20)
]


def ensure_model_downloaded():
    if not os.path.exists(Config.MODEL_PATH):
        urllib.request.urlretrieve(Config.MODEL_URL, Config.MODEL_PATH)


def make_detector(cb):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=Config.MODEL_PATH),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_tracking_confidence=0.45,
        result_callback=cb
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def lm_px(lm, idx, w, h):
    p = lm[idx]
    return int(p.x*w), int(p.y*h)


def relaxed_draw_gesture(lm):
    index_up = lm[INDEX_TIP].y < lm[INDEX_PIP].y
    hand_width = abs(lm[INDEX_MCP].x - lm[PINKY_MCP].x)
    return index_up and hand_width > 0.08


def is_open_hand(lm):
    return (
        lm[8].y < lm[5].y and
        lm[12].y < lm[9].y and
        lm[16].y < lm[13].y and
        lm[20].y < lm[17].y
    )


def smooth_point(prev_pt, new_pt, alpha=0.65):
    if prev_pt is None:
        return new_pt
    x = int(alpha*prev_pt[0] + (1-alpha)*new_pt[0])
    y = int(alpha*prev_pt[1] + (1-alpha)*new_pt[1])
    return (x, y)


def draw_skeleton(frame, lm, w, h):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, lm_px(lm,a,w,h), lm_px(lm,b,w,h), (150,150,150), 1, cv2.LINE_AA)

def build_pixel_templates(cx, cy, radius=95, n=200):
    TAU = math.pi*2
    r = radius

    circle = [
        (int(cx+r*math.cos(TAU*i/n)), int(cy+r*math.sin(TAU*i/n)))
        for i in range(n)
    ]

    corners = [(cx-r,cy-r),(cx+r,cy-r),(cx+r,cy+r),(cx-r,cy+r)]
    square = []
    pps = n//4
    for k in range(4):
        x0,y0 = corners[k]
        x1,y1 = corners[(k+1)%4]
        for j in range(pps):
            t = j/pps
            square.append((int(x0+t*(x1-x0)), int(y0+t*(y1-y0))))

    angs = [270,30,150]
    verts = [
        (int(cx+r*math.cos(a*math.pi/180)), int(cy+r*math.sin(a*math.pi/180)))
        for a in angs
    ]
    triangle = []
    pps = n//3
    for k in range(3):
        x0,y0 = verts[k]
        x1,y1 = verts[(k+1)%3]
        for j in range(pps):
            t = j/pps
            triangle.append((int(x0+t*(x1-x0)), int(y0+t*(y1-y0))))

    return {"circle":circle, "square":square, "triangle":triangle}


def _normalise_resample(pts, n=64):
    if len(pts) < 2:
        return pts

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    cx = (min(xs)+max(xs))/2
    cy = (min(ys)+max(ys))/2
    sc = max((max(xs)-min(xs))/2, (max(ys)-min(ys))/2) or 1

    norm = [((x-cx)/sc, (y-cy)/sc) for x,y in pts]

    while len(norm) < n:
        norm.append(norm[-1])

    idx = np.linspace(0, len(norm)-1, n).astype(int)
    return [norm[i] for i in idx]


NORM_TEMPLATES = {
    k:_normalise_resample(v)
    for k,v in build_pixel_templates(0,0,70).items()
}


def _dtw(a,b):
    return sum(
        math.hypot(a[i][0]-b[i][0], a[i][1]-b[i][1])
        for i in range(len(a))
    ) / len(a)


def compute_score(stroke, shape):
    u = _normalise_resample(stroke)
    t = NORM_TEMPLATES[shape]

    accuracy = max(0, 100 - math.exp(Config.DTW_DECAY_K*_dtw(u,t))*8)
    smoothness = 90.0
    completion = min(100, len(stroke)*1.4)

    final = (
        Config.W_ACCURACY*accuracy +
        Config.W_SMOOTHNESS*smoothness +
        Config.W_COMPLETION*completion
    )

    stars = (
        5 if final >= Config.STAR_5 else
        4 if final >= Config.STAR_4 else
        3 if final >= Config.STAR_3 else
        2 if final >= Config.STAR_2 else
        1
    )

    return {
        "accuracy": round(accuracy,1),
        "smoothness": round(smoothness,1),
        "completion": round(completion,1),
        "final": round(final,1),
        "stars": stars
    }


class DrawingCanvas:
    def __init__(self, h, w):
        self.ink = np.zeros((h,w,3), dtype=np.uint8)
        self.prev = None

    def draw(self, pt):
        if self.prev is not None:
            cv2.line(self.ink, self.prev, pt,
                     Config.INK_COLOR, Config.INK_THICKNESS, cv2.LINE_AA)
        self.prev = pt

    def lift(self):
        self.prev = None

    def clear(self):
        self.ink[:] = 0
        self.prev = None

    def blend(self, frame):
        mask = self.ink.astype(bool).any(axis=2)
        out = frame.copy()
        out[mask] = cv2.addWeighted(frame,0.1,self.ink,0.9,0)[mask]
        return out

@dataclass
class ResultPanel:
    stars:int=0
    message:str=""
    tip:str=""
    scores:dict=field(default_factory=dict)
    visible:bool=False
    ctr:int=0

    def show(self, s, m, t, sc):
        self.stars = s
        self.message = m
        self.tip = t
        self.scores = sc
        self.visible = True
        self.ctr = 0

    def tick(self):
        if self.visible:
            self.ctr += 1
            self.visible = self.ctr <= Config.RESULT_FRAMES

    def render(self, frame):
        if not self.visible:
            return frame

        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (w,h), (15,15,15), -1)
        frame = cv2.addWeighted(frame, 0.30, overlay, 0.70, 0)

        card_w = 700
        card_h = 430
        x1 = (w-card_w)//2
        y1 = (h-card_h)//2 + 25
        x2 = x1 + card_w
        y2 = y1 + card_h

        cv2.rectangle(frame, (x1,y1), (x2,y2), (248,248,248), -1)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (220,220,220), 3)

        cv2.circle(frame, (w//2, y1-5), 34, (130,225,120), -1)
        cv2.circle(frame, (w//2, y1-5), 34, (240,240,240), 3)
        cv2.putText(frame, "*", (w//2-10, y1+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)

        cv2.putText(frame, "Great Job!", (w//2-150, y1+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20,20,60), 3)

        sx = w//2 - 140
        for i in range(5):
            color = (0,200,255) if i < self.stars else (210,210,210)
            cv2.circle(frame, (sx+i*70, y1+150), 22, color, -1)

        bars = [
            ("Accuracy", self.scores["accuracy"], (90,205,90)),
            ("Smoothness", self.scores["smoothness"], (70,150,255)),
            ("Completion", self.scores["completion"], (200,90,220))
        ]

        yy = y1 + 210
        for name, val, col in bars:
            cv2.putText(frame, name, (x1+50, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40,40,40), 2)

            cv2.rectangle(frame, (x1+220, yy-18),
                          (x1+470, yy+8), (225,225,225), -1)

            cv2.rectangle(frame, (x1+220, yy-18),
                          (x1+220+int(val*2.5), yy+8), col, -1)

            cv2.putText(frame, str(int(val)),
                        (x1+500, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

            yy += 52

        cv2.line(frame, (x1+40, y2-95), (x2-40, y2-95), (220,220,220), 2)

        cv2.putText(frame, self.message,
                    (x1+60, y2-58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20,20,60), 2)

        cv2.putText(frame, self.tip,
                    (x1+60, y2-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60,130,60), 2)

        cv2.rectangle(frame, (x2-185, y2-105), (x2-35, y2-10), (235,248,230), -1)
        cv2.rectangle(frame, (x2-185, y2-105), (x2-35, y2-10), (200,220,200), 2)

        cv2.putText(frame, "Final Score",
                    (x2-165, y2-72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 2)

        cv2.putText(frame, str(int(self.scores["final"])),
                    (x2-145, y2-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (40,170,60), 3)

        return frame


def draw_guide(frame, pts, dim=False):
    col = Config.GUIDE_DIM if dim else Config.GUIDE_COLOR
    for i in range(0, len(pts)-1, 2):
        cv2.line(frame, pts[i], pts[i+1], col, 2, cv2.LINE_AA)


def draw_hud(frame, shape, drawing):
    cv2.rectangle(frame, (0,0), (frame.shape[1],50), (20,20,20), -1)

    cv2.putText(frame, f"Shape: {shape}",
                (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,160), 1)

    txt = "Raise Index Finger To Draw" if not drawing else "Drawing..."
    cv2.putText(frame, txt,
                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)


def draw_ai_hud(frame, profile):
    stats = session_summary(profile)
    w = frame.shape[1]

    cv2.rectangle(frame, (w-200,58), (w-10,118), (22,22,22), -1)
    cv2.putText(frame, f"Avg {stats['average']}",
                (w-190,82), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,180), 1)

def main():
    ensure_model_downloaded()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    h, w = frame.shape[:2]
    cx = w//2
    cy = (50+h)//2

    canvas = DrawingCanvas(h,w)
    result = ResultPanel()

    profile = AdaptiveProfile()
    trainer = generate_trainer_state(profile)

    current_shape = "circle"
    current_stroke = []

    open_frames = 0
    was_drawing = False
    draw_on_ctr = 0
    draw_off_ctr = 0
    smooth_tip = None
    last_radius = None
    guides = None

    latest = [None]

    def cb(res, img, ts):
        latest[0] = res

    with make_detector(cb) as detector:
        ts = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame,1)
            ts += 33

            mp_img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            detector.detect_async(mp_img, ts)

            drawing = False
            res = latest[0]

            if res and res.hand_landmarks:
                lm = res.hand_landmarks[0]
                draw_skeleton(frame, lm, w, h)

                tx, ty = lm_px(lm, INDEX_TIP, w, h)
                smooth_tip = smooth_point(smooth_tip, (tx,ty))
                tx, ty = smooth_tip

                if is_open_hand(lm):
                    open_frames += 1
                    if open_frames >= Config.OPEN_HOLD:
                        canvas.clear()
                        current_stroke.clear()
                        result.visible = False
                else:
                    open_frames = 0

                if relaxed_draw_gesture(lm):
                    draw_on_ctr += 1
                    draw_off_ctr = 0
                else:
                    draw_off_ctr += 1
                    draw_on_ctr = 0

                if draw_on_ctr >= Config.DRAW_START_HOLD:
                    drawing = True
                    was_drawing = True
                    current_stroke.append((float(tx),float(ty)))
                    canvas.draw((tx,ty))

                cv2.circle(frame, (tx,ty), 12,
                           (0,220,255) if drawing else (120,120,255), -1)

            if was_drawing and draw_off_ctr >= Config.DRAW_END_HOLD:
                was_drawing = False
                draw_off_ctr = 0
                canvas.lift()

                if len(current_stroke) >= Config.MIN_STROKE_PTS:
                    scores = compute_score(current_stroke, current_shape)
                    profile.update(scores, current_shape)
                    trainer = generate_trainer_state(profile)

                    result.show(
                        scores["stars"],
                        trainer.motivational_text,
                        trainer.focus_tip,
                        scores
                    )

                current_stroke.clear()

            radius = int(Config.GUIDE_RADIUS * trainer.guide_scale)

            if radius != last_radius:
                guides = build_pixel_templates(cx, cy, radius)
                last_radius = radius

            draw_guide(frame, guides[current_shape], result.visible)

            frame = canvas.blend(frame)
            result.tick()
            frame = result.render(frame)

            draw_hud(frame, current_shape, drawing)
            draw_ai_hud(frame, profile)

            cv2.imshow("Kids Air Drawing Modern Final", frame)

            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                break
            elif k == ord("1"):
                current_shape = "circle"
            elif k == ord("2"):
                current_shape = "square"
            elif k == ord("3"):
                current_shape = "triangle"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()