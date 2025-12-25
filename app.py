from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Sentiment API")

model = joblib.load("mvp_sentimientos.joblib")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    text = data.text

    pred = model.predict([text])[0]
    probs = model.predict_proba([text])[0]
    confidence = float(np.max(probs)) * 100

    return {
        "sentiment": pred,
        "confidence": round(confidence, 2)
    }
