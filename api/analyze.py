# -*- coding: utf-8 -*-
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ===============================
# 1) CONFIG – DeepSeek API
# ===============================
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"
API_KEY = "YOUR_API_KEY"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def deepseek(prompt, system=None, temp=0.3):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "deepseek-v3.1",
        "messages": msgs,
        "temperature": temp,
        "max_tokens": 600
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    data = response.json()

    return data["choices"][0]["message"]["content"].strip()


# ===============================
# 2) Symptoms dataset
# ===============================
SYMPTOMS = [
    {"key": "abdominal_pain", "text": "ألم في البطن"},
    {"key": "headache", "text": "صداع"},
    {"key": "nausea", "text": "غثيان"},
    {"key": "dry_mouth", "text": "جفاف الفم"},
    {"key": "fever", "text": "حمى"},
    {"key": "cough", "text": "سعال"},
    {"key": "fatigue", "text": "إرهاق"},
    {"key": "dizziness", "text": "دوخة"},
    {"key": "bloating", "text": "انتفاخ"},
    {"key": "vomiting", "text": "تقيؤ"}
]

SYMPTOM_QUESTIONS = {
    "headache": [
        {
            "question": "ما شدة الصداع في الأيام الماضية؟",
            "options": ["لا يوجد", "خفيف", "متوسط", "شديد", "شديد جدًا"]
        }
    ],
    "abdominal_pain": [
        {
            "question": "ما شدة ألم البطن؟",
            "options": ["لا يوجد", "خفيف", "متوسط", "شديد", "شديد جدًا"]
        }
    ]
}


# ===============================
# 3) Embedding model
# ===============================
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

def detect_symptoms(text, threshold=0.25):
    parts = text.split("و")
    detected = []

    def cosine(a, b):
        return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

    for part in parts:
        emb = model.encode([part])[0]
        sims = [cosine(emb, s) for s in symptom_embeddings]
        idx = int(np.argmax(sims))
        if sims[idx] >= threshold:
            detected.append(SYMPTOMS[idx]["key"])

    return list(set(detected))


# ===============================
# 4) MAIN LOGIC ENTRYPOINT
# ===============================
def process_message(message):
    """
    الدالة اللي يستدعيها n8n
    """
    result = {
        "detected_symptoms": [],
        "questions": {},
        "ai_advice": ""
    }

    # 1) Detect symptoms
    detected = detect_symptoms(message)
    result["detected_symptoms"] = detected

    # 2) Build follow-up questions
    for s in detected:
        if s in SYMPTOM_QUESTIONS:
            result["questions"][s] = SYMPTOM_QUESTIONS[s]

    # 3) If no symptoms → use AI directly
    if len(detected) == 0:
        result["ai_advice"] = deepseek(
            f"المستخدم كتب: {message}\nاعطه إجابة طبية توعوية بسيطة."
        )

    return result


# ===============================
# Example
# ===============================
if __name__ == "__main__":
    text = "حاسس بصداع جامد و شوية غثيان"
    print(json.dumps(process_message(text), ensure_ascii=False, indent=2))
