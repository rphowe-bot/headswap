import os
import uuid
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
from werkzeug.utils import secure_filename

from tracker import analyze_first_frame, render_tracking_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
OUTPUT_DIR = BASE_DIR / "static" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

ALLOWED_VIDEO = {"mp4", "mov", "m4v", "quicktime"}
ALLOWED_IMAGE = {"jpg", "jpeg", "png", "heic", "webp"}


def ext(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def save_file(file, prefix):
    safe = secure_filename(file.filename)
    filename = f"{prefix}_{uuid.uuid4().hex}_{safe}"
    path = UPLOAD_DIR / filename
    file.save(path)
    return path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    video = request.files.get("video")
    face1 = request.files.get("face1")
    face2 = request.files.get("face2")

    if not video or not face1:
        return jsonify({"error": "Video and Face 1 are required."}), 400

    if ext(video.filename) not in ALLOWED_VIDEO:
        return jsonify({"error": "Please upload MP4, MOV, or M4V video."}), 400

    if ext(face1.filename) not in ALLOWED_IMAGE:
        return jsonify({"error": "Face 1 must be JPG, PNG, WEBP, or HEIC."}), 400

    video_path = save_file(video, "video")
    face1_path = save_file(face1, "face1")
    face2_path = None

    if face2 and face2.filename:
        if ext(face2.filename) not in ALLOWED_IMAGE:
            return jsonify({"error": "Face 2 must be JPG, PNG, WEBP, or HEIC."}), 400
        face2_path = save_file(face2, "face2")

    analysis = analyze_first_frame(str(video_path), str(OUTPUT_DIR))

    if "error" in analysis:
        return jsonify(analysis), 400

    session = {
        "video": str(video_path),
        "face1": str(face1_path),
        "face2": str(face2_path) if face2_path else None,
        "first_frame": analysis["first_frame"],
        "faces": analysis["faces"],
    }

    session_id = uuid.uuid4().hex
    session_file = OUTPUT_DIR / f"{session_id}.json"
    session_file.write_text(json.dumps(session), encoding="utf-8")

    return jsonify({
        "session_id": session_id,
        "first_frame_url": url_for("static", filename=f"outputs/{Path(analysis['first_frame']).name}"),
        "faces": analysis["faces"],
        "has_face2": bool(face2_path),
    })


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    target1 = data.get("target1")
    target2 = data.get("target2")

    if not session_id:
        return jsonify({"error": "Missing session."}), 400

    session_file = OUTPUT_DIR / f"{session_id}.json"
    if not session_file.exists():
        return jsonify({"error": "Session expired or missing."}), 400

    session = json.loads(session_file.read_text(encoding="utf-8"))

    if target1 is None:
        return jsonify({"error": "Choose a target face for Face 1."}), 400

    output_name = f"headswap_output_{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_DIR / output_name

    try:
        render_tracking_video(
            video_path=session["video"],
            face1_path=session["face1"],
            face2_path=session.get("face2"),
            initial_faces=session["faces"],
            target1_index=int(target1),
            target2_index=int(target2) if target2 is not None and session.get("face2") else None,
            output_path=str(output_path),
            max_seconds=20,
        )
    except Exception as e:
        return jsonify({"error": f"Render failed: {str(e)}"}), 500

    return jsonify({
        "output_url": url_for("static", filename=f"outputs/{output_name}"),
        "message": "Your tracked meme video is ready."
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
