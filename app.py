from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# FastAPI 앱 초기화
app = FastAPI()

# 모델 및 토크나이저 로드
model_name = "beomi/kcbert-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
model.load_state_dict(torch.load("kc_bert_emotion_classifier.pth", map_location=torch.device('cpu')))
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 요청 바디 모델 정의
class TextRequest(BaseModel):
    text: str

# 예측 함수
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    return predicted_class

# 예측 엔드포인트
@app.post("/predict")
def get_prediction(request: TextRequest):
    try:
        prediction = predict(request.text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
