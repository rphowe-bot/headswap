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
mp_mesh = mp.solutions.face_mesh

# MediaPipe face oval landmark indices
FACE_OVAL_IDX = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109
]


# ── helpers ──────────────────────────────────────────────────────────────────

def read_image_cv(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        pil = Image.open(path).convert("RGBA")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGBA2BGRA)
    return img


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def frame_difference_score(prev_frame, frame):
    if prev_frame is None or frame is None:
        return 0.0
    small_a = cv2.resize(prev_frame, (96, 54), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(frame,      (96, 54), interpolation=cv2.INTER_AREA)
    gray_a = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(gray_a, gray_b)))


# ── face detection ────────────────────────────────────────────────────────────

def get_face_mesh_landmarks(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = []

    with mp_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=4,
        refine_landmarks=True,
        min_detection_confidence=0.60,   # raised from 0.35
        min_tracking_confidence=0.55,    # raised from 0.35
    ) as mesh:
        result = mesh.process(rgb)

    if not result.multi_face_landmarks:
        return faces

    for lm_set in result.multi_face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_set.landmark]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        x  = clamp(min(xs), 0, w - 1)
        y  = clamp(min(ys), 0, h - 1)
        x2 = clamp(max(xs), 0, w - 1)
        y2 = clamp(max(ys), 0, h - 1)
        bw, bh = x2 - x, y2 - y

        # raised threshold: 8% of shorter dimension, minimum 80px
        min_face_size = max(80, int(min(w, h) * 0.08))

        if bw > min_face_size and bh > min_face_size:
            faces.append({
                "x": x, "y": y, "w": bw, "h": bh,
                "cx": x + bw / 2, "cy": y + bh / 2,
                "score": 1.0, "mesh": pts, "source": "mesh",
            })

    faces.sort(key=lambda f: f["cx"])
    return faces


def detect_faces_bgr(frame_bgr, min_confidence=0.65):  # raised from 0.45
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    faces = []

    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=min_confidence) as detector:
        result = detector.process(rgb)

    if not result.detections:
        return faces

    for det in result.detections:
        box  = det.location_data.relative_bounding_box
        x    = int(max(0, box.xmin * w))
        y    = int(max(0, box.ymin * h))
        bw   = int(min(w - x, box.width  * w))
        bh   = int(min(h - y, box.height * h))
        score = float(det.score[0]) if det.score else 0.0

        kps = [(int(kp.x * w), int(kp.y * h))
               for kp in det.location_data.relative_keypoints]

        # raised threshold: 8% of shorter dimension, minimum 80px
        min_face_size = max(80, int(min(w, h) * 0.08))

        if bw > min_face_size and bh > min_face_size:
            faces.append({
                "x": x, "y": y, "w": bw, "h": bh,
                "cx": x + bw / 2, "cy": y + bh / 2,
                "score": score, "keypoints": kps, "source": "detector",
            })

    faces.sort(key=lambda f: f["cx"])
    return faces


def merge_mesh_into_faces(detected_faces, mesh_faces):
    for face in detected_faces:
        best_mesh, best_dist = None, 999999
        for mesh in mesh_faces:
            dist = math.hypot(face["cx"] - mesh["cx"], face["cy"] - mesh["cy"])
            if dist < best_dist:
                best_dist = dist
                best_mesh = mesh
        if best_mesh and best_dist < max(face["w"], face["h"]) * 0.85:
            face["mesh"]   = best_mesh["mesh"]
            face["source"] = "detector+mesh"
    return detected_faces


def detect_faces_cascade(frame_bgr):
    detected  = detect_faces_bgr(frame_bgr, min_confidence=0.65)
    mesh_faces = get_face_mesh_landmarks(frame_bgr)

    if detected:
        return merge_mesh_into_faces(detected, mesh_faces)
    if mesh_faces:
        return mesh_faces
    return []


# ── first-frame analysis ──────────────────────────────────────────────────────

def draw_first_frame(frame, faces, out_path):
    draw = frame.copy()
    for i, f in enumerate(faces):
        x, y, w, h = f["x"], f["y"], f["w"], f["h"]
        color = (255, 200, 50) if i == 0 else (200, 80, 255)
        cv2.rectangle(draw, (x, y), (x + w, y + h), color, 3)
        cv2.putText(draw, f"Face {i + 1}", (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    cv2.imwrite(out_path, draw)


def analyze_first_frame(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video."}

    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total      = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration   = total / fps if fps else 0

    if duration > 21:
        cap.release()
        return {"error": "Video is longer than 20 seconds. Please trim it first."}

    ok, frame = cap.read()
    cap.release()

    if not ok:
        return {"error": "Could not read first frame."}

    faces = detect_faces_cascade(frame)

    if not faces:
        return {"error": "No faces detected on the first frame. Try a clearer starting frame."}

    out_name = f"first_frame_{uuid.uuid4().hex}.jpg"
    out_path = str(Path(output_dir) / out_name)
    draw_first_frame(frame, faces, out_path)

    public_faces = [
        {
            "index": i,
            "x": f["x"], "y": f["y"], "w": f["w"], "h": f["h"],
            "label": f"Face {i + 1}",
            "score": round(f.get("score", 1.0), 3),
        }
        for i, f in enumerate(faces)
    ]

    return {"first_frame": out_path, "faces": public_faces, "duration": duration}


# ── face sticker preparation ──────────────────────────────────────────────────

def make_mesh_mask(w, h, mesh_pts):
    """Precise face-shaped mask using MediaPipe oval landmarks."""
    mask = np.zeros((h, w), dtype=np.uint8)
    try:
        pts = np.array([mesh_pts[i] for i in FACE_OVAL_IDX], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    except Exception:
        # fallback to ellipse if mesh points are unavailable
        cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.42), int(h * 0.50)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def make_soft_oval_mask(w, h):
    """Fallback soft oval mask when no mesh is available."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.42), int(h * 0.50)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask


def auto_crop_face_from_upload(face_bgr):
    if face_bgr is None:
        raise RuntimeError("Could not read uploaded face image.")

    if face_bgr.shape[2] == 4:
        bgr   = face_bgr[:, :, :3]
        alpha = face_bgr[:, :, 3]
    else:
        bgr   = face_bgr[:, :, :3]
        alpha = np.full(face_bgr.shape[:2], 255, dtype=np.uint8)

    gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    detected = cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=4, minSize=(60, 60))

    h, w = bgr.shape[:2]

    if len(detected) > 0:
        detected = sorted(detected, key=lambda r: r[2] * r[3], reverse=True)
        x, y, fw, fh = detected[0]
        pad_x      = int(fw * 0.34)
        pad_top    = int(fh * 0.46)
        pad_bottom = int(fh * 0.36)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_top)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_bottom)
    else:
        side = min(w, h)
        x1   = (w - side) // 2
        y1   = (h - side) // 2
        x2   = x1 + side
        y2   = y1 + side

    crop_bgr   = bgr[y1:y2, x1:x2]
    crop_alpha = alpha[y1:y2, x1:x2]

    if crop_bgr.size == 0:
        side       = min(w, h)
        x1         = (w - side) // 2
        y1         = (h - side) // 2
        crop_bgr   = bgr[y1:y1 + side, x1:x1 + side]
        crop_alpha = alpha[y1:y1 + side, x1:x1 + side]

    return crop_bgr, crop_alpha


def prepare_face_sticker(face_path, target_w, target_h):
    face            = read_image_cv(face_path)
    bgr, alpha      = auto_crop_face_from_upload(face)

    out_w = max(40, int(target_w))
    out_h = max(40, int(target_h))

    bgr   = cv2.resize(bgr,   (out_w, out_h), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask  = make_soft_oval_mask(out_w, out_h)

    alpha = cv2.bitwise_and(alpha.astype(np.uint8), mask.astype(np.uint8))
    sticker = np.dstack([bgr, alpha])
    return sticker, None


def prepare_face_sticker_with_mesh(face_path, target_w, target_h, mesh_pts):
    """Higher quality sticker using mesh-shaped mask when landmarks available."""
    face       = read_image_cv(face_path)
    bgr, alpha = auto_crop_face_from_upload(face)

    out_w = max(40, int(target_w))
    out_h = max(40, int(target_h))

    bgr   = cv2.resize(bgr,   (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    mask  = make_mesh_mask(out_w, out_h, mesh_pts)

    alpha = cv2.bitwise_and(alpha.astype(np.uint8), mask.astype(np.uint8))
    sticker = np.dstack([bgr, alpha])
    return sticker, None


# ── pose estimation ───────────────────────────────────────────────────────────

def estimate_roll(face):
    if "mesh" in face and face["mesh"]:
        pts = face["mesh"]
        try:
            left_eye  = pts[33]
            right_eye = pts[263]
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            return math.degrees(math.atan2(dy, dx))
        except Exception:
            pass

    kps = face.get("keypoints") or []
    if len(kps) >= 2:
        dx = kps[1][0] - kps[0][0]
        dy = kps[1][1] - kps[0][1]
        return math.degrees(math.atan2(dy, dx))

    return 0.0


def get_mesh_anchor(face):
    if "mesh" not in face or not face["mesh"]:
        return face["cx"], face["cy"]
    pts = face["mesh"]
    try:
        left_eye  = pts[33]
        right_eye = pts[263]
        nose      = pts[1]
        return (
            (left_eye[0] + right_eye[0] + nose[0]) / 3,
            (left_eye[1] + right_eye[1] + nose[1]) / 3,
        )
    except Exception:
        return face["cx"], face["cy"]


# ── image transforms ──────────────────────────────────────────────────────────

def rotate_rgba(img, angle_degrees):
    h, w   = img.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    matrix[0, 2] += nw / 2 - center[0]
    matrix[1, 2] += nh / 2 - center[1]
    return cv2.warpAffine(img, matrix, (nw, nh),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def overlay_rgba_poisson(frame, sticker, cx, cy):
    """
    Poisson seamless clone for Hollywood-quality blending.
    Falls back to alpha composite if clone fails.
    """
    h, w   = frame.shape[:2]
    sh, sw = sticker.shape[:2]

    x0 = int(cx - sw / 2)
    y0 = int(cy - sh / 2)
    x1 = x0 + sw
    y1 = y0 + sh

    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return frame

    # Pad sticker if it goes out of frame bounds for seamlessClone
    pad_left   = max(0, -x0)
    pad_top    = max(0, -y0)
    pad_right  = max(0, x1 - w)
    pad_bottom = max(0, y1 - h)

    if pad_left or pad_top or pad_right or pad_bottom:
        sticker = cv2.copyMakeBorder(
            sticker, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
        )
        cx_adj = int(cx)
        cy_adj = int(cy)
    else:
        cx_adj = int(cx)
        cy_adj = int(cy)

    src_bgr  = sticker[:, :, :3]
    src_mask = sticker[:, :, 3]

    # seamlessClone needs mask to be non-zero inside and the center
    # to be strictly inside the destination frame
    cx_safe = clamp(cx_adj, sw // 2 + 1, w - sw // 2 - 1)
    cy_safe = clamp(cy_adj, sh // 2 + 1, h - sh // 2 - 1)

    # Erode mask slightly so edges don't bleed
    kernel      = np.ones((5, 5), np.uint8)
    eroded_mask = cv2.erode(src_mask, kernel, iterations=1)

    try:
        if (eroded_mask > 10).sum() > 100:
            # Resize src to fit exactly in frame region if needed
            region_w = min(sw, w - max(0, cx_safe - sw // 2))
            region_h = min(sh, h - max(0, cy_safe - sh // 2))
            if region_w != sw or region_h != sh:
                src_bgr      = cv2.resize(src_bgr,      (region_w, region_h))
                eroded_mask  = cv2.resize(eroded_mask,  (region_w, region_h))

            result = cv2.seamlessClone(
                src_bgr, frame, eroded_mask,
                (cx_safe, cy_safe),
                cv2.NORMAL_CLONE
            )
            return result
    except Exception:
        pass

    # ── fallback: alpha composite with color match ──
    return _alpha_composite(frame, sticker, cx, cy)


def _alpha_composite(frame, sticker, cx, cy):
    """Original alpha blend with color matching — used as fallback."""
    h, w   = frame.shape[:2]
    sh, sw = sticker.shape[:2]

    x0 = int(cx - sw / 2)
    y0 = int(cy - sh / 2)
    x1 = x0 + sw
    y1 = y0 + sh

    if x1 <= 0 or y1 <= 0 or x0 >= w or y0 >= h:
        return frame

    sx0 = max(0, -x0);  sy0 = max(0, -y0)
    sx1 = sw - max(0, x1 - w)
    sy1 = sh - max(0, y1 - h)
    fx0 = max(0, x0);   fy0 = max(0, y0)
    fx1 = fx0 + (sx1 - sx0)
    fy1 = fy0 + (sy1 - sy0)

    crop = sticker[sy0:sy1, sx0:sx1]
    if crop.size == 0:
        return frame

    fg_rgb = crop[:, :, :3].astype(np.float32)
    alpha  = crop[:, :, 3].astype(np.float32) / 255.0
    alpha  = cv2.GaussianBlur(alpha, (11, 11), 0)[:, :, None]
    roi    = frame[fy0:fy1, fx0:fx1].astype(np.float32)

    fg_mean = np.mean(fg_rgb, axis=(0, 1))
    bg_mean = np.mean(roi,    axis=(0, 1))
    fg_std  = np.std(fg_rgb,  axis=(0, 1))
    bg_std  = np.std(roi,     axis=(0, 1))
    fg_std  = np.where(fg_std < 1, 1, fg_std)

    fg_rgb  = (fg_rgb - fg_mean) / fg_std * bg_std + bg_mean
    fg_rgb  = np.clip(fg_rgb, 0, 255)

    blended = roi * (1 - alpha) + fg_rgb * alpha
    frame[fy0:fy1, fx0:fx1] = blended.astype(np.uint8)
    return frame


# ── tracking helpers ──────────────────────────────────────────────────────────

def match_faces_to_initial(current_faces, initial_faces):
    matches = {}
    for init in initial_faces:
        best_i, best_d = None, float("inf")
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


def smooth_face(face):
    anchor_cx, anchor_cy = get_mesh_anchor(face)
    if "prev_w" not in face:
        face["prev_w"]  = face["w"]
        face["prev_h"]  = face["h"]
        face["prev_cx"] = anchor_cx
        face["prev_cy"] = anchor_cy

    face["prev_w"]  = face["prev_w"]  * 0.86 + face["w"]  * 0.14
    face["prev_h"]  = face["prev_h"]  * 0.86 + face["h"]  * 0.14
    face["prev_cx"] = face["prev_cx"] * 0.84 + anchor_cx  * 0.16
    face["prev_cy"] = face["prev_cy"] * 0.84 + anchor_cy  * 0.16

    return face["prev_cx"], face["prev_cy"], face["prev_w"], face["prev_h"]


def adaptive_scale_for_face(face):
    face_size   = max(face["w"], face["h"], 1)
    size_factor = min(1.0, 130 / face_size)
    return 0.82 + 0.23 * size_factor


def copy_smoothing_state(prev_face, new_face):
    if "prev_w" in prev_face:
        new_face["prev_w"]  = prev_face["prev_w"]
        new_face["prev_h"]  = prev_face["prev_h"]
        new_face["prev_cx"] = prev_face.get("prev_cx", prev_face["cx"])
        new_face["prev_cy"] = prev_face.get("prev_cy", prev_face["cy"])
    return new_face


# ── main render ───────────────────────────────────────────────────────────────

def render_tracking_video(
    video_path, face1_path, face2_path,
    initial_faces, target1_index, target2_index,
    output_path, max_seconds=20
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 720)
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1280)
    max_frames   = int(min(total_frames, fps * max_seconds))

    temp_no_audio = str(Path(output_path).with_suffix(".silent.mp4"))
    fourcc        = cv2.VideoWriter_fourcc(*"mp4v")
    writer        = cv2.VideoWriter(temp_no_audio, fourcc, fps, (width, height))

    initial = [dict(f, cx=f["x"] + f["w"] / 2, cy=f["y"] + f["h"] / 2)
               for f in initial_faces]

    last_target_faces = {}
    previous_frame    = None
    frame_idx         = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        scene_changed = frame_difference_score(previous_frame, frame) > 38.0
        faces         = detect_faces_cascade(frame)
        new_matches   = match_faces_to_initial(faces, initial)

        if scene_changed:
            last_target_faces = {}
            matches = new_matches
        elif last_target_faces:
            locked_matches = {}
            for k, prev_face in last_target_faces.items():
                best, best_dist = None, 999999
                for f in faces:
                    dist       = math.hypot(f["cx"] - prev_face["cx"], f["cy"] - prev_face["cy"])
                    prev_size  = max(prev_face["w"], prev_face["h"], 1)
                    size_ratio = max(f["w"], f["h"], 1) / prev_size
                    if 0.55 <= size_ratio <= 1.85 and dist < best_dist:
                        best_dist = dist
                        best      = f
                if best and best_dist < 56:
                    locked_matches[k] = copy_smoothing_state(prev_face, best)
            matches = locked_matches
        else:
            matches = new_matches

        last_target_faces.update(matches)

        # ── render face 1 ──
        if target1_index in matches:
            f = matches[target1_index]
            smooth_cx, smooth_cy, smooth_w, smooth_h = smooth_face(f)

            scale   = adaptive_scale_for_face(f)
            scale_w = smooth_w * scale
            scale_h = smooth_h * (scale + 0.08)

            # Use mesh-aware sticker if landmarks available
            if "mesh" in f and f["mesh"]:
                sticker, _ = prepare_face_sticker_with_mesh(face1_path, scale_w, scale_h, f["mesh"])
            else:
                sticker, _ = prepare_face_sticker(face1_path, scale_w, scale_h)

            roll    = estimate_roll(f)
            sticker = rotate_rgba(sticker, roll)

            y_offset = smooth_h * 0.12
            frame = overlay_rgba_poisson(frame, sticker, smooth_cx, smooth_cy + y_offset)

        # ── render face 2 ──
        if face2_path and target2_index is not None and target2_index in matches:
            f = matches[target2_index]
            smooth_cx, smooth_cy, smooth_w, smooth_h = smooth_face(f)

            scale   = adaptive_scale_for_face(f)
            scale_w = smooth_w * scale
            scale_h = smooth_h * (scale + 0.08)

            if "mesh" in f and f["mesh"]:
                sticker, _ = prepare_face_sticker_with_mesh(face2_path, scale_w, scale_h, f["mesh"])
            else:
                sticker, _ = prepare_face_sticker(face2_path, scale_w, scale_h)

            roll    = estimate_roll(f)
            sticker = rotate_rgba(sticker, roll)

            y_offset = smooth_h * 0.12
            frame = overlay_rgba_poisson(frame, sticker, smooth_cx, smooth_cy + y_offset)

        writer.write(frame)
        previous_frame = frame.copy()
        frame_idx += 1

    cap.release()
    writer.release()

    # ── ffmpeg: high quality encode with audio ──
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_no_audio,
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "slow",       # better quality than veryfast
        "-crf", "18",            # near-lossless (18=great, 23=default)
        "-profile:v", "high",
        "-level", "4.1",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(temp_no_audio)
    except Exception:
        os.replace(temp_no_audio, output_path)
