import os
import numpy as np
import requests
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ======== KONFIGURASI FOLDER UPLOAD ========
UPLOAD_FOLDER = os.path.join(os.getcwd(), "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ======== KONFIGURASI MODEL ========
MODEL_DIR = os.path.join(os.getcwd(), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "base_model_vgg16.h5")

# Link Google Drive (direct download)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1sUwN4biuZewMX-saeZJ2_YHubcL_pAmf"

# Threshold model
CONF_THRESHOLD = 0.65


def download_model():
    """Download model dari Google Drive jika belum tersedia."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("🔽 Mengunduh model dari Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model berhasil diunduh.")


# ======== LOAD MODEL ========
print("🔍 Mengecek model...")
download_model()
print("📦 Memuat model...")
model = load_model(MODEL_PATH)
print("✅ Model siap digunakan.")


# ======== FUNGSI PREDIKSI ========
def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    # Interpretasi prediksi
    if pred >= CONF_THRESHOLD:
        return "AI", float(pred)
    elif pred <= (1 - CONF_THRESHOLD):
        return "Human", float(1 - pred)
    else:
        return "Bukan lukisan", None


# ======== ROUTE FLASK ========
@app.route("/", methods=["GET", "POST"])
def upload_predict():
    label, confidence, file_path = None, None, None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            label, confidence = predict_single_image(filepath)

            return render_template(
                "index.html",
                file_path=f"/static/uploads/{filename}",
                label=label,
                confidence=confidence,
            )

    return render_template("index.html", file_path=None, label=None, confidence=None)


# ======== ENTRY POINT ========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)