import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename

# Load model and scaler
rf = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Feature extraction function (your original one here)
from your_feature_module import extract_features  # If separated into a file

def predict_image_class(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Invalid image", 0

    features = extract_features(img)
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = rf.predict(features_scaled)[0]
    proba = rf.predict_proba(features_scaled).max()

    return prediction, proba


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

        # Run prediction
        pred_class, confidence = predict_image_class(filepath)

        # return render_template("index.html", image_path=filepath,
        #                        prediction=pred_class,
        #                        confidence=f"{confidence * 100:.2f}%")

        from threading import Timer

        # Function to delete the file after delay
        def delete_file_delayed(path, delay=5):
            def delete():
                if os.path.exists(path):
                    os.remove(path)
            Timer(delay, delete).start()


        # Schedule deletion
        delete_file_delayed(filepath, delay=10)

        # Show the result
        return render_template("index.html", image_path=filepath,
                            prediction=pred_class,
                            confidence=f"{confidence * 100:.2f}%")

    return render_template("index.html", image_path=None)


if __name__ == "__main__":
    app.run(debug=True)


