from flask import Flask, request, jsonify
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# --- DeepSeek Config (من الكود الأصلي) ---
API_URL = "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"
API_KEY = "4_JENf9g9NVi7_332loZt65qIydiAJCPNHhbx0irqaHtJPkfqcUCpp8tp85SlqOU8QX1lYp4AsvLtKqgx0OXRQ"  # ضيفها كـ env var على Vercel

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

def deepseek_chat(prompt, system_prompt=None, max_tokens=512, temperature=0.3):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "deepseek-v3.1",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        return {"error": response.text}

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# --- SYMPTOMS و embeddings (من الكود الأصلي) ---
SYMPTOMS = [  # قائمة الأعراض هنا زي ما في الكود
    {"key": "abdominal_pain", "text": "ألم في البطن"},
    # ... باقي القائمة
]

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
symptom_texts = [s["text"] for s in SYMPTOMS]
symptom_embeddings = model.encode(symptom_texts)

def detect_symptoms_embedding(user_text, top_k=3):
    # الدالة زي ما هي، بس رجع list من dicts
    user_embedding = model.encode([user_text])[0]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    similarities = [cosine_sim(user_embedding, emb) for emb in symptom_embeddings]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    detected = []
    for idx in top_indices:
        detected.append({
            "key": SYMPTOMS[idx]["key"],
            "text": SYMPTOMS[idx]["text"],
            "similarity": similarities[idx]
        })
    return detected

# --- SYMPTOM_QUESTIONS (من الكود الأصلي) ---

# --- API Endpoint ---
@app.route('/api/rag', methods=['GET', 'POST'])
def rag_handler():
    if request.method == 'GET':
        query = request.args.get('query')
    else:
        data = request.json
        query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    # كشف الأعراض
    detected = detect_symptoms_embedding(query)

    # استخدم DeepSeek لو عايز توليد رد إضافي (مثل تفسير)
    deepseek_response = deepseek_chat(f"شرح الأعراض التالية باختصار: {', '.join([d['text'] for d in detected])}")

    # رجع النتيجة كـ JSON
    return jsonify({
        "detected_symptoms": detected,
        "deepseek_explanation": deepseek_response
    })

if __name__ == '__main__':
    app.run()  # للتشغيل محلي، بس Vercel هيحولها serverless
