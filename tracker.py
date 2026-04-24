import os
import uuid
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image


mp_face = mp.solutions.face_detection


def read_image_cv(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        pil = Image.open(path).convert("RGBA")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGRA)
    return img


def detect_faces_bgr(frame_bgr, min_confidence=0.45):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    faces = []
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as detector:
        result = detector.process(rgb)

    if not result.detections:
        return faces

    for det in result.detections:
        box = det.location_data.relative_bounding_box
        x = int(max(0, box.xmin * w))
        y = int(max(0, box.ymin * h))
        bw = int(min(w - x, box.width * w))
        bh = int(min(h - y, box.height * h))
        score = float(det.score[0]) if det.score else 0.0

        kps = []
        for kp in det.location_data.relative_keypoints:
            kps.append((int(kp.x * w), int(kp.y * h)))

        min_face_size = max(35, int(min(w, h) * 0.035))

        if bw > min_face_size and bh > min_face_size:
            faces.append({
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "cx": x + bw / 2,
                "cy": y + bh / 2,
                "score": score,
                "keypoints": kps,
            })

    faces.sort(key=lambda f: f["cx"])
    return faces


def draw_first_frame(frame, faces, out_path):
    draw = frame.copy()
    for i, f in enumerate(faces):
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        color = (255, 200, 50) if i == 0 else (200, 80, 255)
        cv2.rectangle(draw, (x, y), (x + w, y + h), color, 3)
        cv2.putText(draw, f"Face {i+1}", (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, draw)


def analyze_first_frame(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total / fps if fps else 0

    if duration > 21:
        cap.release()
        return {"error": "Video is longer than 20 seconds. Please trim it first."}

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return {"error": "Could not read first frame."}

    faces = detect_faces_bgr(frame)

    if not faces:
        return {"error": "No faces detected on the first frame. Try a clearer starting frame."}

    out_name = f"first_frame_{uuid.uuid4().hex}.jpg"
    out_path = str(Path(output_dir) / out_name)
    draw_first_frame(frame, faces, out_path)

    public_faces = []
    for i, f in enumerate(faces):
        public_faces.append({
            "index": i,
            "x": f["x"],
            "y": f["y"],
            "w": f["w"],
            "h": f["h"],
            "label": f"Face {i+1}",
            "score": round(f["score"], 3),
        })

    return {
        "first_frame": out_path,
        "faces": public_faces,
        "duration": duration,
    }


def make_ripped_mask(w, h):
    """
    Creates a soft oval mask instead of a ripped sticker edge.
    This makes the overlay look cleaner and less like a magazine cutout.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    center = (w // 2, h // 2)
    axes = (int(w * 0.42), int(h * 0.48))

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    # Feather edges for smoother blending
    mask = cv2.GaussianBlur(mask, (21, 21), 0)

def prepare_face_sticker(face_path, target_w, target_h):
    face = read_image_cv(face_path)

    if face.shape[2] == 4:
        bgr = face[:, :, :3]
        alpha = face[:, :, 3]
    else:
        bgr = face[:, :, :3]
        alpha = np.full(face.shape[:2], 255, dtype=np.uint8)

    h, w = bgr.shape[:2]
    side = min(w, h)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    bgr = bgr[y0:y0+side, x0:x0+side]
    alpha = alpha[y0:y0+side, x0:x0+side]

    out_w = max(40, int(target_w))
    out_h = max(40, int(target_h))

    bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_AREA)

    ripped = make_ripped_mask(out_w, out_h)

if ripped.shape != alpha.shape:
    ripped = cv2.resize(ripped, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_AREA)

# Ensure ripped is single channel
if len(ripped.shape) == 3:
    ripped = cv2.cvtColor(ripped, cv2.COLOR_BGR2GRAY)

# Ensure both are uint8
alpha = alpha.astype(np.uint8)
ripped = ripped.astype(np.uint8)

# Final safety resize (just in case)
if ripped.shape != alpha.shape:
    ripped = cv2.resize(ripped, (alpha.shape[1], alpha.shape[0]))

alpha = cv2.bitwise_and(alpha, ripped)

    sticker = np.dstack([bgr, alpha])
    return sticker, None


def rotate_rgba(img, angle_degrees):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    nw = int((h * sin) + (w * cos))
    nh = int((h * cos) + (w * sin))

    matrix[0, 2] += (nw / 2) - center[0]
    matrix[1, 2] += (nh / 2) - center[1]

    return cv2.warpAffine(
        img,
        matrix,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def estimate_roll(face):
    kps = face.get("keypoints") or []
    if len(kps) >= 2:
        right_eye = kps[0]
        left_eye = kps[1]
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]
        return math.degrees(math.atan2(dy, dx))
    return 0.0


def overlay_rgba(frame, sticker, cx, cy):
    h, w = frame.shape[:2]
    sh, sw = sticker.shape[:2]

    x0 = int(cx - sw / 2)
    y0 = int(cy - sh / 2)
    x1 = x0 + sw
    y1 = y0 + sh

    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return frame

    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = sw - max(0, x1 - w)
    sy1 = sh - max(0, y1 - h)

    fx0 = max(0, x0)
    fy0 = max(0, y0)
    fx1 = fx0 + (sx1 - sx0)
    fy1 = fy0 + (sy1 - sy0)

    crop = sticker[sy0:sy1, sx0:sx1]
    if crop.size == 0:
        return frame

    rgb = crop[:, :, :3].astype(np.float32)
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0

    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    frame[fy0:fy1, fx0:fx1] = (roi * (1 - alpha) + rgb * alpha).astype(np.uint8)

    return frame


def match_faces_to_initial(current_faces, initial_faces):
    matches = {}

    for init in initial_faces:
        best_i = None
        best_d = float("inf")

        for i, cur in enumerate(current_faces):
            d = math.hypot(
                cur["cx"] - (init["x"] + init["w"] / 2),
                cur["cy"] - (init["y"] + init["h"] / 2),
            )

            if d < best_d:
                best_d = d
                best_i = i

        if best_i is not None:
            matches[init["index"]] = current_faces[best_i]

    return matches


def smooth_face_size(face):
    if "prev_w" not in face:
        face["prev_w"] = face["w"]
        face["prev_h"] = face["h"]

    face["prev_w"] = face["prev_w"] * 0.7 + face["w"] * 0.3
    face["prev_h"] = face["prev_h"] * 0.7 + face["h"] * 0.3

    return face["prev_w"], face["prev_h"]


def render_tracking_video(video_path, face1_path, face2_path, initial_faces, target1_index, target2_index, output_path, max_seconds=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 720)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1280)

    max_frames = int(min(total_frames, fps * max_seconds))
    temp_no_audio = str(Path(output_path).with_suffix(".silent.mp4"))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_no_audio, fourcc, fps, (width, height))

    initial = []
    for f in initial_faces:
        f2 = dict(f)
        f2["cx"] = f2["x"] + f2["w"] / 2
        f2["cy"] = f2["y"] + f2["h"] / 2
        initial.append(f2)

    last_target_faces = {}
    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        faces = detect_faces_bgr(frame, min_confidence=0.35)
        new_matches = match_faces_to_initial(faces, initial)

        if last_target_faces:
            locked_matches = {}

            for k, prev_face in last_target_faces.items():
                best = None
                best_dist = 99999

                for f in faces:
                    dx = f["cx"] - prev_face["cx"]
                    dy = f["cy"] - prev_face["cy"]
                    dist = (dx * dx + dy * dy) ** 0.5

                    if dist < best_dist:
                        best_dist = dist
                        best = f

                if best and best_dist < 50:
                    locked_matches[k] = best

            matches = locked_matches
        else:
            matches = new_matches

        last_target_faces.update(matches)

        if target1_index in matches:
            f = matches[target1_index]
            smooth_w, smooth_h = smooth_face_size(f)

            scale_w = smooth_w * 1.24
            scale_h = smooth_h * 1.34

            sticker, _ = prepare_face_sticker(face1_path, scale_w, scale_h)
            roll = estimate_roll(f)
            sticker = rotate_rgba(sticker, roll)
            frame = overlay_rgba(frame, sticker, f["cx"], f["cy"])

            cv2.circle(
                frame,
                (int(f["cx"]), int(f["cy"])),
                int(max(f["w"], f["h"]) * 0.72),
                (255, 180, 60),
                2,
            )

        if face2_path and target2_index is not None and target2_index in matches:
            f = matches[target2_index]
            smooth_w, smooth_h = smooth_face_size(f)

            scale_w = smooth_w * 1.24
            scale_h = smooth_h * 1.34

            sticker, _ = prepare_face_sticker(face2_path, scale_w, scale_h)
            roll = estimate_roll(f)
            sticker = rotate_rgba(sticker, roll)
            frame = overlay_rgba(frame, sticker, f["cx"], f["cy"])

            pad = int(max(f["w"], f["h"]) * 0.18)
            cv2.rectangle(
                frame,
                (int(f["x"] - pad), int(f["y"] - pad)),
                (int(f["x"] + f["w"] + pad), int(f["y"] + f["h"] + pad)),
                (230, 90, 255),
                2,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    cmd = [
        "ffmpeg", "-y",
        "-i", temp_no_audio,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(temp_no_audio)
    except Exception:
        os.replace(temp_no_audio, output_path)
