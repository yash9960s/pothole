import os
import cv2
import numpy as np
import joblib
import json
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import torch
import sys
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

# --- PATHS ---
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- FLASK SETUP ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# --- DEVICE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Models (Only once when the server starts) ---
try:
    # YOLO model
    yolo_model = YOLO('best.pt')

    # MiDaS model (choose hybrid or large)
    USE_LARGE = False  # set True if you want dpt-large (slower but more accurate)
    MODEL_NAME = "Intel/dpt-large" if USE_LARGE else "Intel/dpt-hybrid-midas"

    processor = DPTImageProcessor.from_pretrained(MODEL_NAME)
    midas_model = DPTForDepthEstimation.from_pretrained(MODEL_NAME).to(device).eval()

    # Regression model
    reg_model = joblib.load("regression_model.pkl")
    with open("regression_features.json") as f:
        feature_order = json.load(f)["feature_order"]

    print("All models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: A required model file was not found. Please check your project folder.")
    print(f"Details: {e}")
    sys.exit()

# --- HELPER FUNCTIONS ---
def impact_category(depth, width, length):
    two_wheeler = "low"
    four_wheeler = "low"
    size = depth + width + length
    
    if size < 50:
        two_wheeler = "low"
        four_wheeler = "low"
    elif size < 100:
        two_wheeler = "moderate"
        four_wheeler = "low"
    elif size < 150:
        two_wheeler = "severe"
        four_wheeler = "moderate"
    else:
        two_wheeler = "extreme"
        four_wheeler = "severe"
        
    return two_wheeler, four_wheeler

def run_midas(frame):
    """Run Hugging Face MiDaS on an OpenCV frame and return depth map"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth = outputs.predicted_depth

    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

    return depth_resized

def process_frame(frame):
    # Process with YOLOv8
    results = yolo_model(frame)
    annotated_frame = results[0].plot()
    
    # Collect all pothole data
    pothole_info_list = []
    
    # Run MiDaS depth
    depth_map = run_midas(frame)

    for r in results:
        boxes = r.boxes
        if not boxes:
            continue
            
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf.item())

            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            # MiDaS Depth Estimation
            pothole_area_depth = depth_map[y1:y2, x1:x2]
            if pothole_area_depth.size == 0:
                continue
            
            midas_bbox_mean = float(pothole_area_depth.mean())
            midas_scene_median = float(np.median(depth_map))
            midas_rel = midas_bbox_mean - midas_scene_median

            # Regression Model Prediction
            features_dict = {
                "confidence": confidence,
                "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                "midas_bbox_mean": midas_bbox_mean,
                "midas_scene_median": midas_scene_median,
                "midas_rel": midas_rel
            }
            
            features = np.array([[features_dict[f] for f in feature_order]])
            pred_depth, pred_width, pred_length = reg_model.predict(features)[0]

            # Impact Estimation
            two_wheeler_impact, four_wheeler_impact = impact_category(pred_depth, pred_width, pred_length)

            pothole_info_list.append({
                "depth": pred_depth, 
                "width": pred_width, 
                "length": pred_length,
                "2W_impact": two_wheeler_impact,
                "4W_impact": four_wheeler_impact
            })

    # --- Add text output ---
    y_offset = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    line_spacing = 30

    for i, info in enumerate(pothole_info_list):
        label = (f"Pothole {i+1}: D:{info['depth']:.1f}cm, W:{info['width']:.1f}cm, "
                 f"L:{info['length']:.1f}cm, 2W:{info['2W_impact']}, 4W:{info['4W_impact']}")
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(annotated_frame, (10, y_offset - text_height - 5),
                      (10 + text_width, y_offset + baseline), (0, 0, 0), -1)
        
        cv2.putText(annotated_frame, label, (10, y_offset), font, font_scale, (0, 255, 0), font_thickness)
        y_offset += line_spacing
            
    return annotated_frame, pothole_info_list

# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    video_url = url_for('video_feed')
    uploaded_image = None
    txt_output = None
    pothole_data = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = Path(app.config['UPLOAD_FOLDER']) / filename
            file.save(str(filepath))
            file_stem = filepath.stem

            img = cv2.imread(str(filepath))
            processed_img, pothole_data = process_frame(img)

            annotated_filename = 'annotated_' + filename
            annotated_path = Path(app.config['UPLOAD_FOLDER']) / annotated_filename
            cv2.imwrite(str(annotated_path), processed_img)
            
            uploaded_image = annotated_filename
            
            txt_filename = file_stem + '.txt'
            txt_path = Path(app.config['UPLOAD_FOLDER']) / txt_filename
            with open(str(txt_path), 'w') as f:
                if pothole_data:
                    for i, info in enumerate(pothole_data):
                        f.write(f"Pothole {i+1}:\n")
                        f.write(f"  - Dimensions: D:{info['depth']:.1f}cm, W:{info['width']:.1f}cm, L:{info['length']:.1f}cm\n")
                        f.write(f"  - 2W Impact: {info['2W_impact']}\n")
                        f.write(f"  - 4W Impact: {info['4W_impact']}\n\n")
                else:
                    f.write("No potholes detected in this image.\n")
            txt_output = txt_filename
            
    return render_template('index.html', uploaded_image=uploaded_image, txt_output=txt_output, video_url=video_url)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            return

        while True:
            success, frame = camera.read()
            if not success:
                break
            
            processed_frame, _ = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        camera.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    app.run(debug=True)
