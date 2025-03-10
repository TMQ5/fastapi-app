from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

# تحميل نموذج KMeans ومعيار التحجيم
kmeans = joblib.load("kmeans_makeup.joblib")
scaler = joblib.load("scaler.joblib")

# إنشاء التطبيق
app = FastAPI()

# تعريف نموذج بيانات الإدخال
class InputData(BaseModel):
    features: list[float]

# نقطة نهاية لـ API للتنبؤ باستخدام النموذج
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # تحويل البيانات إلى مصفوفة NumPy
        features_array = np.array(data.features).reshape(1, -1)

        # تطبيق التحجيم باستخدام StandardScaler
        scaled_features = scaler.transform(features_array)

        # التنبؤ باستخدام نموذج KMeans
        prediction = kmeans.predict(scaled_features)

        return {"cluster": int(prediction[0])}
    
    except Exception as e:
        return {"error": str(e)}

# نقطة نهاية للوصول إلى الوثائق التفاعلية
@app.get("/")
def root():
    return {"message": "API is running! Access /docs for interactive UI."}
