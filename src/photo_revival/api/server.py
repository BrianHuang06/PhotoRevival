# -*- coding: utf-8 -*-
"""
PhotoRevive Backend API Server
Flask REST API for old photo restoration
"""

import os
import sys
import uuid
import json
import time
import threading
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
import numpy as np

from photo_revival.paths import (
    API_DATA_DIR,
    BASE_MODELS_DIR,
    CHECKPOINTS_DIR,
    CLASSIFIERS_DIR,
    PROJECT_ROOT,
    WEIGHTS_DIR,
)

BASE_DIR = str(PROJECT_ROOT)
UPLOAD_DIR = str(API_DATA_DIR / "uploads")
RESULT_DIR = str(API_DATA_DIR / "results")
HISTORY_FILE = str(API_DATA_DIR / "history.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__, static_folder=None)
CORS(app, origins=["http://localhost:5173"])

pipeline = None
pipeline_lock = threading.Lock()
task_status = {}

SDXL_MODEL_PATH = str(
    BASE_MODELS_DIR
    / "AI-ModelScope"
    / "stable-diffusion-xl-1___0-inpainting-0___1"
)

LORA_SEARCH_PATHS = [
    str(CHECKPOINTS_DIR / "lora_inpainting_v29" / "epoch5"),
    str(CHECKPOINTS_DIR / "lora_inpainting_v28" / "best"),
    str(CHECKPOINTS_DIR / "lora_inpainting_v27" / "best"),
    str(CHECKPOINTS_DIR / "lora_inpainting_v26" / "final"),
    str(CHECKPOINTS_DIR / "lora_inpainting_v25" / "best"),
]

SWINIR_SEARCH_PATHS = [
    str(WEIGHTS_DIR / "realesrgan_s4_swinir_100k.pth"),
    str(CHECKPOINTS_DIR / "swinir" / "best_model.pth"),
]


def find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline

    lora_path = find_file(LORA_SEARCH_PATHS)
    swinir_path = find_file(SWINIR_SEARCH_PATHS)

    from photo_revival.two_stage_restoration import TwoStageRestorationPipeline

    print("[API] Loading models (this may take a while)...")
    pipeline = TwoStageRestorationPipeline(
        sdxl_model_path=SDXL_MODEL_PATH,
        lora_path=lora_path,
        swinir_model_path=swinir_path,
        device="auto",
    )
    pipeline.load()
    print("[API] All models loaded!")
    return pipeline


def run_restore_task(task_id, image_path, params):
    try:
        task_status[task_id]["status"] = "detecting"
        task_status[task_id]["progress"] = 10
        save_task_to_history(task_status[task_id])

        p = get_pipeline()

        image = Image.open(image_path).convert("RGB")
        original_size = image.size

        task_status[task_id]["status"] = "processing"
        task_status[task_id]["progress"] = 20
        save_task_to_history(task_status[task_id])

        mode = params.get("mode", "quick")

        if mode == "quick":
            s1_strength = 0.6
            s2_strength = 0.1
        elif mode == "fine":
            s1_strength = min(1.0, max(0.3, params.get("strength", 0.6) + 0.2))
            s2_strength = params.get("strength", 0.1)
        else:
            s1_strength = 0.5
            s2_strength = 0.05

        task_status[task_id]["progress"] = 30

        results = p.run(
            image,
            stage1_strength=s1_strength,
            stage2_strength=s2_strength,
            use_swinir=True,
            seed=params.get("seed", 42),
        )

        task_status[task_id]["progress"] = 90

        result_img = results.get("final", results.get("stage2_enhanced", image))
        mask_img = results.get("scratch_mask", None)
        stage1_img = results.get("swinir", None)

        result_filename = f"{task_id}_restored.jpg"
        result_path = os.path.join(RESULT_DIR, result_filename)
        result_img.save(result_path, "JPEG", quality=95)

        mask_filename = None
        if mask_img:
            mask_filename = f"{task_id}_mask.png"
            mask_img.save(os.path.join(RESULT_DIR, mask_filename))

        stage1_filename = None
        if stage1_img:
            stage1_filename = f"{task_id}_stage1.jpg"
            stage1_img.save(os.path.join(RESULT_DIR, stage1_filename), "JPEG", quality=95)

        task_status[task_id]["status"] = "completed"
        task_status[task_id]["progress"] = 100
        task_status[task_id]["result_url"] = f"/api/images/{result_filename}"
        task_status[task_id]["mask_url"] = f"/api/images/{mask_filename}" if mask_filename else None
        task_status[task_id]["stage1_url"] = f"/api/images/{stage1_filename}" if stage1_filename else None
        task_status[task_id]["completed_at"] = datetime.now().isoformat()
        save_task_to_history(task_status[task_id])

    except Exception as e:
        print(f"[API] Error in task {task_id}: {traceback.format_exc()}")
        task_status[task_id]["status"] = "error"
        task_status[task_id]["error"] = str(e)
        save_task_to_history(task_status[task_id])


def save_task_to_history(task_data):
    history = load_history()
    found = False
    for i, h in enumerate(history):
        if h.get("id") == task_data.get("id"):
            history[i] = task_data
            found = True
            break
    if not found:
        history.insert(0, task_data)
    history = history[:100]
    save_history(history)


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return jsonify({"error": "Unsupported file type"}), 400

    task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    ext = os.path.splitext(f.filename)[1]
    filename = f"{task_id}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)
    f.save(filepath)

    img = Image.open(filepath)
    w, h = img.size

    task_info = {
        "id": task_id,
        "filename": f.filename,
        "original_url": f"/api/images/{filename}",
        "mode": "quick",
        "status": "idle",
        "progress": 0,
        "created_at": datetime.now().isoformat(),
        "width": w,
        "height": h,
        "file_size": os.path.getsize(filepath),
        "params": {
            "mode": "quick",
            "strength": 0.35,
            "guidanceScale": 7.5,
            "steps": 25,
            "seed": 42,
            "blendFactor": 0.8,
        },
    }

    task_status[task_id] = task_info
    save_task_to_history(task_info)

    return jsonify(task_info)


@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.json or {}
    task_id = data.get("task_id", "")
    sensitivity = data.get("sensitivity", 0.5)

    if task_id not in task_status:
        return jsonify({"error": "Task not found"}), 404

    task = task_status[task_id]
    original_url = task.get("original_url", "")
    filename = original_url.split("/")[-1]
    image_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(image_path):
        return jsonify({"error": "Image file not found"}), 404

    try:
        classifier_path = str(CLASSIFIERS_DIR / "scratch_classifier_v2.pkl")
        if os.path.exists(classifier_path):
            from photo_revival.learn_scratch_v2 import ImprovedMaskDetector
            detector = ImprovedMaskDetector(classifier_path)
            img = Image.open(image_path).convert("RGB")
            img_resized = img.resize((512, 512), Image.LANCZOS)
            mask = detector.detect(np.array(img_resized), stride=4, threshold=0.3)
        else:
            from photo_revival.two_stage_restoration import ScratchDetector
            detector = ScratchDetector(sensitivity=sensitivity)
            img = Image.open(image_path).convert("RGB")
            img_resized = img.resize((512, 512), Image.LANCZOS)
            mask = detector.detect(np.array(img_resized))

        mask_filename = f"{task_id}_detect_mask.png"
        mask_path = os.path.join(RESULT_DIR, mask_filename)
        Image.fromarray(mask).save(mask_path)

        coverage = float(np.mean(mask) / 255 * 100)

        return jsonify({
            "mask_url": f"/api/images/{mask_filename}",
            "coverage": round(coverage, 2),
        })

    except Exception as e:
        print(f"[API] Detection error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/restore", methods=["POST"])
def restore():
    data = request.json or {}
    task_id = data.get("task_id", "")

    if task_id not in task_status:
        return jsonify({"error": "Task not found"}), 404

    task = task_status[task_id]
    if task["status"] in ("detecting", "processing"):
        return jsonify({"error": "Task already in progress"}), 409

    params = {
        "mode": data.get("mode", task.get("params", {}).get("mode", "quick")),
        "strength": data.get("strength", 0.35),
        "guidanceScale": data.get("guidanceScale", 7.5),
        "steps": data.get("steps", 25),
        "seed": data.get("seed", 42),
        "blendFactor": data.get("blendFactor", 0.8),
    }

    task["params"] = params
    task["mode"] = params["mode"]
    task["status"] = "pending"
    task["progress"] = 5

    original_url = task.get("original_url", "")
    filename = original_url.split("/")[-1]
    image_path = os.path.join(UPLOAD_DIR, filename)

    thread = threading.Thread(
        target=run_restore_task,
        args=(task_id, image_path, params),
        daemon=True,
    )
    thread.start()

    return jsonify({"task_id": task_id, "status": "processing"})


@app.route("/api/task/<task_id>", methods=["GET"])
def get_task(task_id):
    if task_id in task_status:
        return jsonify(task_status[task_id])

    history = load_history()
    for h in history:
        if h.get("id") == task_id:
            return jsonify(h)

    return jsonify({"error": "Task not found"}), 404


@app.route("/api/history", methods=["GET"])
def get_history():
    history = load_history()
    for task_id, task in task_status.items():
        found = False
        for h in history:
            if h.get("id") == task_id:
                found = True
                break
        if not found:
            history.insert(0, task)

    result = []
    for h in history[:50]:
        item = {
            "id": h.get("id", ""),
            "filename": h.get("filename", ""),
            "original_url": h.get("original_url", ""),
            "restored_url": h.get("restored_url") or h.get("result_url"),
            "mask_url": h.get("mask_url"),
            "mode": h.get("mode", "quick"),
            "status": h.get("status", "idle"),
            "created_at": h.get("created_at", ""),
            "width": h.get("width"),
            "height": h.get("height"),
        }
        result.append(item)

    return jsonify({"tasks": result})


@app.route("/api/history/<task_id>", methods=["DELETE"])
def delete_history(task_id):
    history = load_history()
    history = [h for h in history if h.get("id") != task_id]
    save_history(history)

    if task_id in task_status:
        del task_status[task_id]

    return jsonify({"success": True})


@app.route("/api/images/<filename>", methods=["GET"])
def serve_image(filename):
    for d in [RESULT_DIR, UPLOAD_DIR]:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            return send_file(path, mimetype="image/jpeg")

    return jsonify({"error": "Image not found"}), 404


@app.route("/api/status", methods=["GET"])
def server_status():
    return jsonify({
        "status": "ok",
        "pipeline_loaded": pipeline is not None,
        "active_tasks": sum(
            1 for t in task_status.values()
            if t.get("status") in ("detecting", "processing")
        ),
        "cuda_available": __import__("torch").cuda.is_available(),
    })


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    dist_dir = os.path.join(BASE_DIR, "frontend", "dist")
    if not os.path.exists(dist_dir):
        return jsonify({
            "message": "PhotoRevive API is running. Build frontend with: cd frontend && npm run build",
            "api_docs": "/api/status",
        })

    if path and os.path.exists(os.path.join(dist_dir, path)):
        return send_from_directory(dist_dir, path)

    return send_from_directory(dist_dir, "index.html")


if __name__ == "__main__":
    print("=" * 50)
    print("PhotoRevive API Server")
    print("=" * 50)

    history = load_history()
    for h in history:
        h["status"] = "idle"
        h["progress"] = 0
        task_status[h["id"]] = h
    print(f"[API] Loaded {len(task_status)} tasks from history")

    print("[API] Starting server on http://localhost:5000")
    print("[API] Frontend dev server should be on http://localhost:5173")
    print("[API] Models will be loaded on first restore request")
    print()

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
