import requests
import json

# عنوان الـ API
url = "http://127.0.0.1:5001/predict_symptoms"

# بيانات الطلب (الأعراض)
data = {
    "symptoms": ["itching", "skin_rash"]
}

# إرسال طلب POST
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json.dumps(data), headers=headers)

# طباعة الاستجابة
print("Status Code:", response.status_code)
print("Response:", response.json())