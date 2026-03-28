import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import base64


# BACKGROUND IMAGE STYLE

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(
                rgba(0, 0, 0, 0.65),
                rgba(0, 0, 0, 0.65)
            ),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("new.png")


# APP CONFIG

st.set_page_config(
    page_title="AI-Powered Mental Health Detection",
    layout="centered"
)
st.markdown(
    """
    <h2 style="
        font-size: 32px;
        font-weight: 600;
        margin-bottom: 10px;
    ">
        AI-Powered Mental Health Detection
    </h2>
    """,
    unsafe_allow_html=True
)
st.write(
    "This application uses a fine-tuned **DistilBERT NLP model** "
    "to detect mental health signals from text."
)
st.markdown(
    """
    <h1 style='
        text-align: center;
        font-size: 90px;
        font-weight: 800;
        margin-top: 20px;
    '>
        PsyDetect
    </h1>
    """,
    unsafe_allow_html=True
)
st.warning("⚠️ This tool is for educational purposes only and NOT a medical diagnosis.")


# LABELS

LABELS = ["Normal", "Depression", "Anxiety", "Suicidal"]


# KEYWORD SIGNALS

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
    text = text.lower()
    return sum(1 for w in keywords if w in text)


# LOAD MODEL & TOKENIZER 
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained(
        "mental_health_distilbert_best"
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "mental_health_distilbert_best"
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device

model, tokenizer, device = load_model()


# USER INPUT

user_text = st.text_area(
    "Enter a sentence or social media post:",
    placeholder="I feel exhausted and nothing seems meaningful anymore...",
    height=160
)


# ANALYSIS

if st.button("Analyze"):
    if user_text.strip() == "":
        st.error("Please enter some text to analyze.")
    else:
        inputs = tokenizer(
            user_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        top2 = torch.topk(probs, 2)
        p1, p2 = top2.indices.tolist()
        c1, c2 = probs[p1].item(), probs[p2].item()

        suicidal_risk = (probs[3] > 0.30) or (symptom_score(user_text, SUICIDAL_WORDS) > 0)

        
        # RESULTS
        
        st.subheader("Prediction Result")
        st.success(
            f"**Primary Prediction:** {LABELS[p1]} ({c1:.2f})\n\n"
            f"**Secondary Prediction:** {LABELS[p2]} ({c2:.2f})"
        )

        st.subheader("Confidence Scores")
        for i, label in enumerate(LABELS):
            st.write(f"{label}: {probs[i].item():.2f}")

        
# FOOTER

st.markdown("---")
st.caption("Final Year Project | B.Sc. Applied Data Science & Communication")

