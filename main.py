from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# ✅ **تحميل البيانات النظيفة لاستخراج الترميزات**
try:
    products_data = pd.read_csv("products_data.csv")

    # ✅ **حساب التشفير الترددي (Frequency Encoding)**
    brand_freq_encoding = products_data['brand_name'].value_counts() / len(products_data)
    makeup_freq_encoding = products_data['makeup_type'].value_counts() / len(products_data)

except Exception as e:
    raise RuntimeError(f"❌ خطأ أثناء تحميل البيانات: {str(e)}")

# ✅ **تحميل نموذج KMeans ومعيار التحجيم**
try:
    kmeans = joblib.load("kmeans_makeup.joblib")
    scaler = joblib.load("scaler.joblib")
except Exception as e:
    raise RuntimeError(f"❌ خطأ أثناء تحميل النماذج: {str(e)}")

# ✅ **إنشاء التطبيق**
app = FastAPI()

# ✅ **قاموس تحويل أرقام الكلستر إلى الفئات النصية المحددة من قبل المستخدم**
cluster_mapping = {
    0: "أساسيات اقتصادية",
    1: "أساسيات وأدوات اقتصادية",
    2: "أساسيات باهظة الثمن",
    3: "أساسيات منخفضة التكلفة"
}

# ✅ **نموذج بيانات الإدخال**
class FeaturesInput(BaseModel):
    brand_name: str
    makeup_type: str
    original_price: float
    discounted_price: float
    reviews_number: int

@app.post("/predict/")
async def predict_cluster(input_data: FeaturesInput):
    try:
        # ✅ **تحويل البيانات الفئوية إلى عددية باستخدام Frequency Encoding**
        brand_encoded = brand_freq_encoding.get(input_data.brand_name, 0.0)  # إذا لم يكن موجودًا، يتم وضع 0
        makeup_encoded = makeup_freq_encoding.get(input_data.makeup_type, 0.0)  # إذا لم يكن موجودًا، يتم وضع 0

        # ✅ **إنشاء مصفوفة الميزات (features)**
        features = [brand_encoded, makeup_encoded, input_data.original_price, input_data.discounted_price, input_data.reviews_number]

        # ✅ **تحجيم الميزات العددية**
        scaled_features = scaler.transform([features])

        # ✅ **التنبؤ باستخدام نموذج KMeans**
        prediction = kmeans.predict(scaled_features)

        # ✅ **تحويل الرقم إلى الفئة النصية المناسبة**
        cluster_label = cluster_mapping.get(int(prediction[0]), "غير محدد")

        return {
            "brand": input_data.brand_name,
            "makeup_type": input_data.makeup_type,
            "predicted_category": cluster_label  # ✅ الآن يتم إرجاع التصنيف النصي المحدد من قبل المستخدم
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ خطأ أثناء التنبؤ: {str(e)}")

# ✅ **نقطة نهاية للوصول إلى الوثائق التفاعلية**
@app.get("/")
def root():
    return {"message": "🚀 API جاهز ويعمل! يمكنك زيارة /docs لاستخدام الواجهة التفاعلية."}
