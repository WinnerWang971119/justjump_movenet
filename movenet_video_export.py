
import argparse
import time
import csv
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# 17-keypoint names in COCO order (MoveNet uses this)
KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON_EDGES = [
    ("left_hip","right_hip"), ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
    ("left_hip","left_knee"), ("left_knee","left_ankle"),
    ("right_hip","right_knee"), ("right_knee","right_ankle"),
    ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
    ("nose","left_eye"), ("left_eye","left_ear"),
    ("nose","right_eye"), ("right_eye","right_ear"),
    ("nose","left_shoulder"), ("nose","right_shoulder"),
]

def build_model(variant: str):
    variant = variant.lower()
    if variant == "lightning":
        url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        input_size = 192
    elif variant == "thunder":
        url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        input_size = 256
    else:
        raise ValueError("variant must be 'lightning' or 'thunder'")
    model = hub.load(url)
    return model.signatures["serving_default"], input_size

def preprocess_frame(frame_bgr, input_size):
    """Resize to (input_size,input_size), float32 [0,1], add batch dim."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    x = tf.convert_to_tensor(resized, dtype=tf.float32) / 255.0
    x = tf.expand_dims(x, axis=0)
    return x

def run_movenet(movenet_fn, input_tensor):
    outputs = movenet_fn(input_tensor)
    # output_0: shape [1,1,17,3] => y, x, confidence (all normalized 0..1 for x,y)
    kps = outputs["output_0"].numpy()[0, 0, :, :]  # (17,3)
    return kps

def draw_skeleton(frame, kps_xyc, score_thresh=0.3):
    h, w, _ = frame.shape
    # draw points
    for (name, (x, y, c)) in kps_xyc.items():
        if c < score_thresh: continue
        cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)
    # draw edges
    for a,b in SKELETON_EDGES:
        xa, ya, ca = kps_xyc[a]
        xb, yb, cb = kps_xyc[b]
        if ca >= score_thresh and cb >= score_thresh:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255,255,255), 1)
    return frame

def export_csv(csv_path, rows):
    header = ["frame_idx","time_ms","keypoint","x","y","score","x_norm","y_norm"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def export_jsonl(jsonl_path, per_frame_objects):
    import json
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in per_frame_objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser(description="MoveNet video keypoint exporter (single person).")
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument("--outdir", default="movenet_out", help="Output directory")
    parser.add_argument("--variant", default="lightning", choices=["lightning","thunder"], help="Model variant")
    parser.add_argument("--min-score", type=float, default=0.3, help="Min confidence to draw/save")
    parser.add_argument("--preview", action="store_true", help="Show a preview window while processing")
    parser.add_argument("--save-preview", action="store_true", help="Save a preview mp4 with skeleton overlay")
    parser.add_argument("--skip", type=int, default=0, help="Number of frames to skip between inferences (0 = every frame)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Outputs
    csv_path = outdir / "keypoints.csv"
    jsonl_path = outdir / "keypoints.jsonl"
    rows = []
    jsonl_frames = []

    if args.save_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_path = outdir / "preview.mp4"
        vw = cv2.VideoWriter(str(preview_path), fourcc, fps, (width, height))
    else:
        vw = None

    movenet_fn, input_size = build_model(args.variant)

    # Warmup
    dummy = tf.zeros([1, input_size, input_size, 3], dtype=tf.float32)
    _ = run_movenet(movenet_fn, dummy)

    frame_idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Optional skipping
        do_infer = True
        if args.skip > 0 and (frame_idx % (args.skip + 1) != 0):
            do_infer = False

        time_ms = int((frame_idx / fps) * 1000.0)

        if do_infer:
            inp = preprocess_frame(frame, input_size)
            kps = run_movenet(movenet_fn, inp)  # (17,3) y,x,score normalized

            # Convert to pixel coords
            per_frame_obj = {"frame_idx": frame_idx, "time_ms": time_ms, "keypoints": {}}
            kps_xyc = {}
            for i, name in enumerate(KEYPOINTS):
                y_norm, x_norm, score = kps[i]
                x_px = x_norm * width
                y_px = y_norm * height
                per_frame_obj["keypoints"][name] = {
                    "x": float(x_px), "y": float(y_px),
                    "x_norm": float(x_norm), "y_norm": float(y_norm),
                    "score": float(score),
                }
                rows.append([frame_idx, time_ms, name, x_px, y_px, score, x_norm, y_norm])
                kps_xyc[name] = (x_px, y_px, score)

            jsonl_frames.append(per_frame_obj)

            if args.preview or args.save_preview:
                overlay = frame.copy()
                draw_skeleton(overlay, kps_xyc, score_thresh=args.min_score)
                if args.preview:
                    cv2.imshow("MoveNet Preview", overlay)
                    # Press q to quit early
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if vw is not None:
                    vw.write(overlay)
        else:
            if args.preview:
                cv2.imshow("MoveNet Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_idx += 1

    cap.release()
    if vw is not None:
        vw.release()
    cv2.destroyAllWindows()

    export_csv(csv_path, rows)
    export_jsonl(jsonl_path, jsonl_frames)

    elapsed = time.time() - t0
    print(f"Done. Frames processed: {frame_idx}/{total}. Time: {elapsed:.2f}s")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")
    if args.save_preview:
        print(f"Preview video: {outdir / 'preview.mp4'}")

if __name__ == "__main__":
    main()
