# PsyDetect — AI-Powered Mental Health Detection

⚠️ **Disclaimer:** This tool is for **educational and research purposes only**. It is **not** a medical diagnostic tool and should **not** replace professional clinical assessment or mental health support.

---

## About

**PsyDetect** is a Streamlit-based web application developed as a Final Year Project for a B.Sc. in Applied Data Science & Communication. It uses a fine-tuned **DistilBERT** transformer model to classify text input into one of four mental health categories, providing confidence scores for each prediction.

The goal of this project is to explore how Natural Language Processing (NLP) can be applied to support early detection of mental health signals in text — such as social media posts or written expressions — in an academic and research context.


## Features

- **Text Classification** — Classifies input into: `Normal`, `Depression`, `Anxiety`, or `Suicidal`
- **Confidence Scoring** — Returns a confidence score for each category
- **Primary & Secondary Predictions** — Detects overlapping emotional states
- **Simple Web Interface** — Clean, accessible UI built with Streamlit
- **Real-time Inference** — Instant results powered by a fine-tuned transformer model

---

## Project Structure

```
web/
├── step_1.py                    # Data preprocessing
├── step_2.py                    # Model training
├── step_3_test_model.py         # Model evaluation
├── step_4_app.py                # Streamlit web application
├── preprocessed_data_balanced.pt  # Preprocessed dataset
├── new.png                      # Background image
└── README.md


 Categories 

| Label | Description |
|---|---|
| Normal | No significant mental health signals detected |
| Depression | Signals of low mood, hopelessness, or emotional fatigue |
| Anxiety | Signals of persistent worry, fear, or restlessness |
| Suicidal | Signals of suicidal ideation or self-harm language |


##  Ethical Considerations

This project handles sensitive mental health data and language. The following principles were followed during development:

- The model is trained and evaluated on publicly available, labelled research datasets
- All outputs are probabilistic estimates, not clinical diagnoses
- The application clearly communicates its educational-only purpose to users


## Author

**Shainy Ravishika Ruparthna** — B.Sc. Applied Data Science & Communication
Final Year Project | 2026
