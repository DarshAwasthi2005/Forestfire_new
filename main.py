from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import numpy as np
import shap

application = FastAPI(title="Forest Fire Prediction API")
ridge_model = pickle.load(open('Algorithms/ridge_model.pkl','rb'))
Standard_model = pickle.load(open('Algorithms/scaler.pkl','rb'))

feature_names = [
    'Date', 'Month', 'Year', 'Temperature', 'RH', 'Ws',
    'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region'
]

explainer = shap.LinearExplainer(ridge_model, masker=np.zeros((1, len(feature_names))))

@application.get("/", response_class=HTMLResponse)
def home():
    return "Send POST to /predict"

@application.post("/predict")
def predict(
    Date: float = Form(...), Month: float = Form(...), Year: float = Form(...),
    Temperature: float = Form(...), RH: float = Form(...), Ws: float = Form(...),
    Rain: float = Form(...), FFMC: float = Form(...), DMC: float = Form(...),
    ISI: float = Form(...), Classes: float = Form(...), Region: float = Form(...)
):
    new_data = Standard_model.transform([[Date, Month, Year, Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    prediction = ridge_model.predict(new_data)
    return {"prediction": float(prediction[0])}


@application.post("/explain")
def explain_prediction(data: dict):
    input_array = np.array([[data[feat] for feat in feature_names]])
    prediction = ridge_model.predict(input_array)[0]
    shap_values = explainer.shap_values(input_array)
    feature_impact = sorted(
        zip(feature_names, shap_values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return {
        "prediction": float(prediction),
        "top_3_features": [
            {"feature": feat, "impact": round(float(val), 3)}
            for feat, val in feature_impact[:3]
        ],
        "all_features": {feat: round(float(val), 3) for feat, val in feature_impact}
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)