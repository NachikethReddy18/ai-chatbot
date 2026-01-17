from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import json
import random
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# -------------------------
# APP SETUP
# -------------------------
app = FastAPI(
    title="AI Customer Support Chatbot",
    version="1.0.0",
    description="Intent-based customer support chatbot with real-world responses"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# LOAD MODEL
# -------------------------
MODEL_DIR = "models/intent_distilbert"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f"{MODEL_DIR}/label_map.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

CONFIDENCE_THRESHOLD = 0.6

# -------------------------
# REQUEST / RESPONSE SCHEMAS
# -------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    intent: str
    confidence: float
    escalate: bool
    reply: str
    action: str

# -------------------------
# RESPONSE DATA
# -------------------------
ORDER_LOCATIONS = ["Hyderabad", "Bangalore", "Chennai", "Mumbai", "Pune"]
DELIVERY_TIMES = ["today", "tomorrow", "within 2 days", "in the next 48 hours"]

ORDER_RESPONSES = [
    "📦 Your order is currently in **{location}** and should arrive **{eta}**.",
    "🚚 Good news! Your package has reached **{location}** and will be delivered **{eta}**.",
    "📍 I checked your order — it’s in **{location}** and expected **{eta}**.",
    "📦 Your shipment is moving smoothly and is presently in **{location}**. Delivery expected **{eta}**."
]

REFUND_RESPONSES = [
    "💸 Your refund has been initiated and will be credited within **5–7 business days**.",
    "✅ We’ve started processing your refund. You should receive the amount within **a week**.",
    "💳 Refund confirmed. The amount will be credited back shortly."
]

PAYMENT_RESPONSES = [
    "💳 It looks like the payment didn’t go through. Please retry or use a different method.",
    "⚠️ We noticed a payment issue. Retrying after a few minutes usually helps.",
    "🔄 Payment failed due to a temporary issue. Please try again."
]

TECH_RESPONSES = [
    "🛠 Our technical team is aware of this issue and is actively working on it.",
    "⚙️ Engineers are investigating the problem. A fix is on the way.",
    "🚧 Thanks for reporting this. The issue has been forwarded to our technical team."
]

ANALYTICS_RESPONSE = (
    "📊 I can help with order statistics, but I don’t have access to your account data yet. "
    "Once account integration is enabled, you’ll be able to view monthly order counts."
)

DEFAULT_RESPONSES = [
    "🤖 I understand your concern and I’m here to help.",
    "🙌 Thanks for reaching out. Let me assist you further.",
    "📞 I’ll make sure your query gets the right attention."
]

# -------------------------
# HELPERS
# -------------------------
def is_analytics_question(message: str) -> bool:
    analytics_keywords = [
        "how many", "number of", "count", "total",
        "this month", "last month", "so far"
    ]
    msg = message.lower()
    return any(k in msg for k in analytics_keywords)

# -------------------------
# RESPONSE ENGINE
# -------------------------
def generate_response(intent: str, message: str):
    msg = message.lower()

    # Analytics / history questions
    if is_analytics_question(msg):
        return {
            "reply": ANALYTICS_RESPONSE,
            "action": "account_analytics"
        }

    # Order tracking
    if "order" in msg:
        return {
            "reply": random.choice(ORDER_RESPONSES).format(
                location=random.choice(ORDER_LOCATIONS),
                eta=random.choice(DELIVERY_TIMES)
            ),
            "action": "order_tracking"
        }

    # Refunds
    if "refund" in msg:
        return {
            "reply": random.choice(REFUND_RESPONSES),
            "action": "refund_process"
        }

    # Payments
    if "payment" in msg:
        return {
            "reply": random.choice(PAYMENT_RESPONSES),
            "action": "payment_support"
        }

    # Technical issues
    if any(word in msg for word in ["app", "error", "issue", "crash"]):
        return {
            "reply": random.choice(TECH_RESPONSES),
            "action": "technical_support"
        }

    # Default fallback
    return {
        "reply": random.choice(DEFAULT_RESPONSES),
        "action": "human_support"
    }

# -------------------------
# HEALTH CHECK
# -------------------------
@app.get("/")
def health():
    return {"status": "API running"}

# -------------------------
# CHAT ENDPOINT
# -------------------------
@app.post("/api/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    inputs = tokenizer(
        req.message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    idx = int(torch.argmax(probs))
    intent = id2label[idx]
    confidence = float(probs[idx])
    escalate = confidence < CONFIDENCE_THRESHOLD

    response = generate_response(intent, req.message)

    return {
        "intent": intent,
        "confidence": round(confidence, 4),
        "escalate": escalate,
        "reply": response["reply"],
        "action": response["action"]
    }
