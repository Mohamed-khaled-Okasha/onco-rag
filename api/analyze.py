# api/analyze.py
import os
import re
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from mangum import Mangum

app = FastAPI(title="Oncology RAG API")

# CORS عشان n8n و Postman يشتغلوا من أي مكان
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepSeek Huawei Cloud
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"
API_KEY = os.getenv("DEEPSEEK_API_KEY")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# تحميل النموذج مرة واحدة
print("جاري تحميل نموذج الـ embeddings...")
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"}, {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"}, {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"}, {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"}, {"key": "dizziness", "text": "دوخة"},
    {"key": "voice_changes", "text": "تغيرات في جودة الصوت"}, {"key": "hoarseness", "text": "بحة الصوت"},
    {"key": "taste_changes", "text": "تغير الطعم"}, {"key": "low_appetite", "text": "انخفاض الشهية"},
    {"key": "vomiting", "text": "تقيؤ"}, {"key": "heartburn", "text": "حرقة صدر"},
    {"key": "gas", "text": "الغازات"}, {"key": "bloating", "text": "الانتفاخ"},
    {"key": "hiccups", "text": "زغطة"}, {"key": "constipation", "text": "امساك"},
    {"key": "diarrhea", "text": "اسهال"}, {"key": "fecal_incontinence", "text": "سلس برازي"},
    {"key": "breath_shortness", "text": "ضيق تنفس"},
]

symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

def detect_symptoms(text, threshold=0.19):
    detected = set()
    parts = re.split(r"[،,.\s!؟؛]+", text.lower())
    for part in parts:
        part = part.strip()
        if len(part) < 3: continue
        user_emb = model.encode([part])[0]
        similarities = [np.dot(user_emb, emb) / (np.linalg.norm(user_emb) * np.linalg.norm(emb)) 
                       for emb in symptom_embeddings]
        for idx, sim in enumerate(similarities):
            if sim > threshold:
                detected.add(SYMPTOMS[idx]["key"])
    return list(detected)

def deepseek_chat(prompt):
    payload = {
        "model": "deepseek-v3.1",
        "messages": [
            {"role": "system", "content": "أنت طبيب أورام متخصص، جاوب بالعربي الفصحى وباختصار."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.3
    }
    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=40)
        return r.json()["choices"][0]["message"]["content"].strip() if r.status_code == 200 else "خطأ في الـ AI"
    except:
        return "مشكلة مؤقتة في الاتصال بالذكاء الاصطناعي"

class Input(BaseModel):
    symptoms: str
    patient_phone: str = ""

@app.post("/api/analyze")
async def analyze(data: Input):
    text = data.symptoms.strip()
    if not text:
        return {"error": "الأعراض فارغة"}

    detected = detect_symptoms(text)
    
    if not detected:
        response = "لم أتعرف على أعراض واضحة، ممكن توضح أكتر؟"
        risk = "منخفضة"
    else:
        symptoms_ar = "، ".join([s["text"] for s in SYMPTOMS if s["key"] in detected])
        prompt = f"المريض يقول: {text}\nالأعراض المكتشفة: {symptoms_ar}\n\nجاوب بالعربي فقط:\n1. الاحتمالات الطبية\n2. درجة الخطورة (منخفضة/متوسطة/عالية/طوارئ)\n3. التوصية الفورية"
        response = deepseek_chat(prompt)
        risk = "طوارئ" if any(w in response for w in ["طوارئ","فورًا","مستشفى","نزيف"]) else \
               "عالية" if any(w in response for w in ["عالية","خطير"]) else "متوسطة"

    return {
        "medical_response": response,
        "detected_symptoms": detected,
        "risk_level": risk,
        "patient_phone": data.patient_phone
    }

# Vercel handler
handler = Mangum(app, lifespan="off")
