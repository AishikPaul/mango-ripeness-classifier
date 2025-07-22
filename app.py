import os
from flask import Flask, render_template, request, redirect
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from threading import Timer
from ultralytics import YOLO

from your_feature_module import extract_features  # Or paste directly if inline

# Load models
rf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
yolo_model = YOLO("best.pt")  # Adjust as needed

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def check_and_classify_mango(image_path, conf_thresh=0.8):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, "Image not found."

    results = yolo_model(image_path, conf=conf_thresh, verbose=False)
    detections = results[0].boxes

    if detections is None or len(detections) == 0:
        return None, None, "No mango detected."

    # Use first detection
    box = detections[0].xyxy.cpu().numpy().astype(int)[0]
    x1, y1, x2, y2 = box
    cropped_img = img[y1:y2, x1:x2]

    features = extract_features(cropped_img)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = rf.predict(features_scaled)[0]
    confidence = rf.predict_proba(features_scaled).max()

    return prediction, confidence, None


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Run YOLO + Classification
        pred_class, confidence, error_msg = check_and_classify_mango(filepath)

        def delete_file_delayed(path, delay=10):
            def delete():
                if os.path.exists(path):
                    os.remove(path)
            Timer(delay, delete).start()

        delete_file_delayed(filepath)

        return render_template(
            "index.html",
            image_path=filepath,
            prediction=pred_class,
            confidence=f"{confidence * 100:.2f}%" if confidence else None,
            error=error_msg
        )

    return render_template("index.html", image_path=None)
if __name__ == "__main__":
    app.run(debug=True)
