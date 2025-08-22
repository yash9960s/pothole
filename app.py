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

# --- PATHS ---
MIDAS_PATH = 'MiDaS'
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Add MiDaS to system path to allow imports
sys.path.append(MIDAS_PATH)

# --- MiDaS Imports ---
try:
    from midas.dpt_depth import DPTDepthModel
    from midas.transforms import Resize, NormalizeImage, PrepareForNet
    import torchvision.transforms as transforms
except ImportError as e:
    print(f"Error importing MiDaS modules. Make sure the 'MiDaS' folder is in your project directory.")
    print(f"Details: {e}")
    sys.exit()

# --- FLASK SETUP ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)

# --- DEVICE SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Load Models (Only once when the server starts) ---
try:
    yolo_model = YOLO('best.pt')
    midas_model = DPTDepthModel(path="MiDaS/dpt_hybrid_384.pt", backbone="vitb_rn50_384", non_negative=True)
    midas_model.eval().to(device)
    reg_model = joblib.load("regression_model.pkl")
    with open("regression_features.json") as f:
        feature_order = json.load(f)["feature_order"]
    print("All models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: A required model file was not found. Please check your project folder.")
    print(f"Details: {e}")
    sys.exit()

midas_transform = transforms.Compose([
    Resize(384, 384, resize_target=None, keep_aspect_ratio=True,
           ensure_multiple_of=32, resize_method="upper_bound",
           image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet()
])

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

def process_frame(frame):
    # Process with YOLOv8
    results = yolo_model(frame)
    annotated_frame = results[0].plot()
    
    # List to collect all pothole data for display
    pothole_info_list = []
    
    # Prepare image for MiDaS
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    input_batch = midas_transform({"image": img_rgb})["image"]
    input_batch = torch.from_numpy(input_batch).unsqueeze(0).to(device)

    # Get depth map
    with torch.no_grad():
        depth_map = midas_model(input_batch)
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    for r in results:
        boxes = r.boxes
        if not boxes:
            continue
            
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf.item())

            # Check if bounding box is valid
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            # --- MiDaS Depth Estimation & Feature Extraction ---
            pothole_area_depth = depth_map[y1:y2, x1:x2]
            if pothole_area_depth.size == 0:
                continue
            
            midas_bbox_mean = float(pothole_area_depth.mean())
            midas_scene_median = float(np.median(depth_map))
            midas_rel = midas_bbox_mean - midas_scene_median

            # --- Regression Model Prediction ---
            features_dict = {
                "confidence": confidence,
                "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                "midas_bbox_mean": midas_bbox_mean,
                "midas_scene_median": midas_scene_median,
                "midas_rel": midas_rel
            }
            
            features = np.array([[features_dict[f] for f in feature_order]])
            pred_depth, pred_width, pred_length = reg_model.predict(features)[0]

            # --- Impact Estimation ---
            two_wheeler_impact, four_wheeler_impact = impact_category(pred_depth, pred_width, pred_length)

            # --- Collect data for text output ---
            pothole_info_list.append({
                "depth": pred_depth, 
                "width": pred_width, 
                "length": pred_length,
                "2W_impact": two_wheeler_impact,
                "4W_impact": four_wheeler_impact
            })

    # --- Add all collected info to top-left of the image ---
    y_offset = 40 # Starting position for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8 # Larger font size for better readability
    font_thickness = 2
    line_spacing = 30 # Spacing between lines

    for i, info in enumerate(pothole_info_list):
        label = (f"Pothole {i+1}: D:{info['depth']:.1f}cm, W:{info['width']:.1f}cm, L:{info['length']:.1f}cm, "
                 f"2W Impact:{info['2W_impact']}, 4W Impact:{info['4W_impact']}")
        
        # Add a black background for readability
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(annotated_frame, (10, y_offset - text_height - 5), (10 + text_width, y_offset + baseline), (0, 0, 0), -1)
        
        # Add the text
        cv2.putText(annotated_frame, label, (10, y_offset), font, font_scale, (0, 255, 0), font_thickness)
        y_offset += line_spacing # Increase offset for the next line
            
    return annotated_frame, pothole_info_list
# --- FLASK ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    video_url = url_for('video_feed')
    
    # Initialize variables with a default value
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

            # Process the uploaded image
            img = cv2.imread(str(filepath))
            processed_img, pothole_data = process_frame(img)

            # Save the annotated image
            annotated_filename = 'annotated_' + filename
            annotated_path = Path(app.config['UPLOAD_FOLDER']) / annotated_filename
            cv2.imwrite(str(annotated_path), processed_img)
            
            uploaded_image = annotated_filename
            
            # --- Save the pothole data to a TXT file ---
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
            
            # Process the frame with the full pipeline
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