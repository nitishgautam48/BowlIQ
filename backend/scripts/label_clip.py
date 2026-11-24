"""Label a clip for the regression suite: step through it frame by frame and set the ground truth.

    python scripts/label_clip.py tests/clips/my_clip.mp4

Keys (click the window first):
    d / →        next frame              a / ←        previous frame
    D / e        +10 frames              A / q        −10 frames
    b            set BFC here            f            set FFC here            r            set release here
    v            cycle view (side-on → front-on → behind → oblique)
    m            toggle bowling arm      o            toggle direction (or none for front-on/behind)
    t            cycle tolerance (1–4 frames) for all three moments
    s            save the sidecar JSON next to the video      x / Esc      quit (asks to save if changed)

Writes <video>.json: {"file", "view", "bowling_arm", "direction", "bfc_frame", "ffc_frame", "release_frame",
"tolerance_frames", "slow_motion", "height_cm", "notes"}. Frame numbers are those of the video as it is played here;
for variable-frame-rate phone clips run the clip through BowlIQ once and label the processed copy
(storage/analyses/<id>/side_prep.mp4) so the numbering matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

VIEWS = ["side-on", "front-on", "behind", "oblique"]
DIRS = ["left_to_right", "right_to_left", None]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--height", type=float, default=None, help="bowler height in cm (optional)")
    ap.add_argument("--slow", type=float, default=None, help="slow-motion factor if the clip is slowed down")
    a = ap.parse_args()
    video = Path(a.video)
    side = video.with_suffix(".json")
    lab = json.loads(side.read_text()) if side.exists() else {
        "file": video.name, "view": "side-on", "bowling_arm": "right", "direction": "left_to_right",
        "bfc_frame": None, "ffc_frame": None, "release_frame": None,
        "tolerance_frames": {"bfc": 2, "ffc": 2, "release": 1}, "slow_motion": a.slow, "height_cm": a.height,
        "labelled_by": "label_clip.py", "notes": ""}
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if n <= 0:
        print("Can't read the video.")
        return 1
    frames: dict[int, any] = {}

    def frame(i: int):
        if i not in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            frames[i] = img if ok else None
            if len(frames) > 400:
                frames.pop(next(iter(frames)))
        return frames[i]

    i, changed = 0, False
    win = "BowlIQ label_clip"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        img = frame(i)
        if img is None:
            i = max(0, i - 1)
            continue
        img = img.copy()
        h = img.shape[0]
        s = max(0.5, h / 720)
        lines = [f"frame {i}/{n - 1}  t={i / fps:.3f}s  ({fps:.2f} fps)",
                 f"BFC {lab['bfc_frame']}  FFC {lab['ffc_frame']}  release {lab['release_frame']}"
                 f"  tol ±{lab['tolerance_frames']['bfc']}",
                 f"view {lab['view']}  arm {lab['bowling_arm']}  dir {lab['direction']}" + ("  *unsaved*" if changed else "")]
        for k, t in enumerate(lines):
            cv2.putText(img, t, (10, int((28 + 28 * k) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), int(4 * s))
            cv2.putText(img, t, (10, int((28 + 28 * k) * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 255, 255), int(2 * s))
        for key, col in (("bfc_frame", (255, 128, 0)), ("ffc_frame", (0, 0, 255)), ("release_frame", (0, 200, 0))):
            if lab[key] == i:
                cv2.rectangle(img, (0, 0), (img.shape[1] - 1, h - 1), col, int(8 * s))
        cv2.imshow(win, img)
        k = cv2.waitKeyEx(0)
        c = chr(k & 0xFF) if (k & 0xFF) < 128 else ""
        if k in (2555904, 65363) or c == "d":
            i = min(n - 1, i + 1)
        elif k in (2424832, 65361) or c == "a":
            i = max(0, i - 1)
        elif c in ("D", "e"):
            i = min(n - 1, i + 10)
        elif c in ("A", "q"):
            i = max(0, i - 10)
        elif c in ("b", "f", "r"):
            lab[{"b": "bfc_frame", "f": "ffc_frame", "r": "release_frame"}[c]] = i
            changed = True
        elif c == "v":
            lab["view"] = VIEWS[(VIEWS.index(lab["view"]) + 1) % len(VIEWS)] if lab["view"] in VIEWS else VIEWS[0]
            changed = True
        elif c == "m":
            lab["bowling_arm"] = "left" if lab["bowling_arm"] == "right" else "right"
            changed = True
        elif c == "o":
            lab["direction"] = DIRS[(DIRS.index(lab["direction"]) + 1) % len(DIRS)]
            changed = True
        elif c == "t":
            t = lab["tolerance_frames"]["bfc"] % 4 + 1
            lab["tolerance_frames"] = {"bfc": t, "ffc": t, "release": max(1, t - 1)}
            changed = True
        elif c == "s":
            side.write_text(json.dumps(lab, indent=2))
            changed = False
            print(f"Saved {side}")
        elif c == "x" or k == 27:
            if changed and input("Save changes? [y/N] ").strip().lower() == "y":
                side.write_text(json.dumps(lab, indent=2))
                print(f"Saved {side}")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
