from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# ---------- Disease Analyzer imports ----------
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# =================================================
# 1️⃣ PRICE PREDICTION (UNCHANGED)
# =================================================

df = pd.read_csv("karnataka_crop_price_till_2025.csv")

le_crop = LabelEncoder()
le_district = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["District"] = le_district.fit_transform(df["District"])

X = df[["Crop", "District", "Year", "Month"]]
y = df["Price_INR_per_Quintal"]

model = LinearRegression()
model.fit(X, y)


@app.route("/api/predict-price", methods=["POST"])
def predict_price():
    data = request.get_json()

    crop = data.get("crop")
    district = data.get("district")
    year = int(data.get("year"))
    month = int(data.get("month"))

    crop_enc = le_crop.transform([crop])[0]
    district_enc = le_district.transform([district])[0]

    future_rows = []

    for i in range(1, 4):
        m = month + i
        y_ = year

        if m > 12:
            m -= 12
            y_ += 1

        future_rows.append([crop_enc, district_enc, y_, m])

    future_df = pd.DataFrame(
        future_rows,
        columns=["Crop", "District", "Year", "Month"]
    )

    predicted_quintal = model.predict(future_df)

    predicted_kg = [round(p / 100, 2) for p in predicted_quintal]

    return jsonify({
        "month1_price": predicted_kg[0],
        "month2_price": predicted_kg[1],
        "month3_price": predicted_kg[2]
    })


# =================================================
# 2️⃣ DISEASE ANALYZER (FIXED)
# =================================================

DISEASE_MODEL_PATH = "model/plant_disease_model.h5"

# 🔧 CRITICAL FIX IS HERE
disease_model = load_model(DISEASE_MODEL_PATH, compile=False)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DISEASES = [
    ("Leaf Spot", "Remove infected leaves and spray neem oil."),
    ("Powdery Mildew", "Apply sulfur-based fungicide."),
    ("Blight", "Use recommended fungicide and remove affected plants."),
    ("Mosaic Virus", "Remove infected plants and control insects."),
    ("Healthy", "No disease detected. Maintain crop care.")
]


@app.route("/api/analyze-disease", methods=["POST"])
def analyze_disease():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files["image"]
    img_path = os.path.join(UPLOAD_FOLDER, img_file.filename)
    img_file.save(img_path)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = disease_model.predict(img_array)

    index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    disease, treatment = DISEASES[index % len(DISEASES)]

    return jsonify({
        "disease": disease,
        "treatment": treatment,
        "confidence": round(confidence, 2)
    })


# =================================================
# ENTRY POINT
# =================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
