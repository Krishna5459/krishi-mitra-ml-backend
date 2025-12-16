from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# ======================================
# LOAD DATASET & TRAIN MODEL
# ======================================

df = pd.read_csv("karnataka_crop_price_till_2025.csv")

le_crop = LabelEncoder()
le_district = LabelEncoder()

df["Crop"] = le_crop.fit_transform(df["Crop"])
df["District"] = le_district.fit_transform(df["District"])

X = df[["Crop", "District", "Year", "Month"]]
y = df["Price_INR_per_Quintal"]

model = LinearRegression()
model.fit(X, y)

# ======================================
# PRICE PREDICTION API
# ======================================

@app.route("/api/predict-price", methods=["POST"])
def predict_price():
    data = request.get_json()

    crop = data["crop"]
    district = data["district"]
    year = int(data["year"])
    month = int(data["month"])

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


# ======================================
# ENTRY POINT
# ======================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
