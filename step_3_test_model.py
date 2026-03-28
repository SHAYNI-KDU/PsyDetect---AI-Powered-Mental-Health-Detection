import torch
import re
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


MODEL_PATH = "mental_health_distilbert_best"
LABELS = ["Normal", "Depression", "Anxiety", "Suicidal"]
MAX_LEN = 128


# DEVICE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# LOAD MODEL & TOKENIZER

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()


# LIGHT CLEAN (same as training)

def light_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# SYMPTOM KEYWORDS

DEPRESSION_WORDS = [
    "hopeless", "empty", "worthless", "tired",
    "no energy", "nothing matters", "numb", "lonely"
]

ANXIETY_WORDS = [
    "worried", "panic", "fear", "nervous",
    "overthinking", "heart racing", "anxious", "scared"
]

SUICIDAL_WORDS = [
    "want to die", "kill myself", "end my life",
    "no reason to live", "better off dead"
]

def symptom_score(text, keywords):
    return sum(1 for w in keywords if w in text)


# SMART PREDICTION FUNCTION

def predict(text):
    text = light_clean(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    # Top-2 predictions
    top2 = torch.topk(probs, 2)
    p1, p2 = top2.indices.tolist()
    c1, c2 = probs[p1].item(), probs[p2].item()

    # Symptom checks
    dep = symptom_score(text, DEPRESSION_WORDS)
    anx = symptom_score(text, ANXIETY_WORDS)
    sui = symptom_score(text, SUICIDAL_WORDS)

    suicidal_risk = (probs[3] > 0.30) or (sui > 0)

    return {
        "primary": (LABELS[p1], c1),
        "secondary": (LABELS[p2], c2),
        "suicidal_risk": suicidal_risk,
        "all_probs": probs.cpu().numpy()
    }


# CLI TEST

if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or type exit): ")
        if text.lower() == "exit":
            break

        result = predict(text)

        print("\nPrimary:", result["primary"])
        print("Secondary:", result["secondary"])

        if result["suicidal_risk"]:
            print("⚠️ Suicidal risk detected")

