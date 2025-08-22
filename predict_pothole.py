import torch
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import joblib
import sys

# --- PATHS ---
YOLO_PATH = 'yolov5'
MIDAS_PATH = 'MiDaS'
IMAGES_FOLDER = 'testing'
OUTPUT_FOLDER = 'outputs/predictions'

sys.path.append(YOLO_PATH)
sys.path.append(MIDAS_PATH)

# --- YOLOv5 ---
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# --- MiDaS ---
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import torchvision.transforms as transforms

# --- DEVICE ---
device = select_device('0' if torch.cuda.is_available() else 'cpu')

# --- Load YOLOv5 ---
weights_yolo = 'best.pt'  # your custom trained model
yolo_model = DetectMultiBackend(weights_yolo, device=device, dnn=False)
stride, pt = yolo_model.stride, yolo_model.pt

# --- Load class names safely ---
if hasattr(yolo_model.model, 'names'):
    names = yolo_model.model.names
else:
    names = {0: 'pothole'}  # fallback

# --- Load MiDaS ---
midas_model = DPTDepthModel(
    path="MiDaS/dpt_large_384.pt",
    backbone="vitl16_384",
    non_negative=True
)
midas_model.eval()
midas_model.to(device)

transform = transforms.Compose([
    Resize(384, 384, resize_target=None, keep_aspect_ratio=True,
           ensure_multiple_of=32, resize_method="upper_bound",
           image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
    PrepareForNet()
])

# --- Load regression model & feature schema ---
reg_model = joblib.load("outputs/regression_model.pkl")
with open("outputs/regression_features.json") as f:
    feature_order = json.load(f)["feature_order"]

# --- Impact helper ---
def impact_category(depth, width, length):
    size = depth + width + length
    if size < 50:
        return "low", "low"
    elif size < 100:
        return "moderate", "low"
    elif size < 150:
        return "severe", "moderate"
    else:
        return "extreme", "severe"

# --- Prepare output folder ---
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

# --- Process each image ---
for img_path in Path(IMAGES_FOLDER).glob("*.jpg"):
    img0 = cv2.imread(str(img_path))
    
    # YOLO detection
    im = letterbox(img0, 640, stride=stride, auto=True)[0]
    im = im.transpose((2,0,1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float() / 255.0
    im = im.unsqueeze(0)

    pred = yolo_model(im)
    pred = non_max_suppression(pred, 0.25, 0.45)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det.cpu().numpy():
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(conf)
                class_id = int(cls)
                class_name = names.get(class_id, f'class_{class_id}')

                # --- Skip non-potholes ---
                if class_name != 'pothole':
                    continue

                # --- MiDaS Depth ---
                img_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)/255.0
                input_batch = transform({"image": img_rgb})["image"]
                input_batch = torch.from_numpy(input_batch).unsqueeze(0).to(device)

                with torch.no_grad():
                    depth_map = midas_model(input_batch)
                    depth_map = torch.nn.functional.interpolate(
                        depth_map.unsqueeze(1),
                        size=img0.shape[:2],
                        mode="bicubic",
                        align_corners=False
                    ).squeeze().cpu().numpy()

                # Extract MiDaS features
                midas_bbox_mean = float(depth_map[y1:y2, x1:x2].mean())
                midas_scene_median = float(np.median(depth_map))
                midas_rel = midas_bbox_mean - midas_scene_median

                # --- Regression features in correct order ---
                features_dict = {
                    "confidence": confidence,
                    "x_min": x1,
                    "y_min": y1,
                    "x_max": x2,
                    "y_max": y2,
                    "midas_bbox_mean": midas_bbox_mean,
                    "midas_scene_median": midas_scene_median,
                    "midas_rel": midas_rel
                }
                features = np.array([[features_dict[f] for f in feature_order]])

                # --- Regression prediction ---
                pred_depth, pred_width, pred_length = reg_model.predict(features)[0]

                # --- Impacts ---
                two_wheeler, four_wheeler = impact_category(pred_depth, pred_width, pred_length)

                # --- Output message ---
                msg = (f"{img_path.name}: There is a pothole ahead, "
                       f"estimated depth: {pred_depth:.1f}, width: {pred_width:.1f}, length: {pred_length:.1f}, "
                       f"impact for two-wheeler: {two_wheeler}, impact for four-wheeler: {four_wheeler}")
                print(msg)

                # --- Annotate image ---
                cv2.rectangle(img0, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(img0, f"{pred_depth:.1f},{pred_width:.1f},{pred_length:.1f}",
                            (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                # --- Save text prediction ---
                txt_path = f"{OUTPUT_FOLDER}/{img_path.stem}.txt"
                with open(txt_path, "w") as f:
                    f.write(msg)

    # Save annotated image
    cv2.imwrite(f"{OUTPUT_FOLDER}/{img_path.stem}_annotated.jpg", img0)

print("✅ Prediction complete. Check annotated images and txt files in outputs/predictions")
