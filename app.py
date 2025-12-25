from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Sentiment API")

model = joblib.load("mvp_sentimientos.joblib")

@app.post("/predict")
def predict(text: str):
    pred = model.predict([text])[0]

    # Probabilidades
    probs = model.predict_proba([text])[0]
    confidence = float(np.max(probs)) * 100

    return {
        "sentiment": pred,
        "confidence": round(confidence, 2)
    }
