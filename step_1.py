import pandas as pd
import numpy as np
import re
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast


# 1. LOAD DATA

train_data = pd.read_csv("mental_heath_unbanlanced.csv")

print("Original dataset shape:", train_data.shape)
print(train_data.head(10))

train_data = train_data[['text', 'status']]


# 2. LIGHT TEXT CLEANING (BERT FRIENDLY)
#clean only what is necessary and don’t remove grammar, punctuation, or context, because BERT relies on natural language understanding.
#Heavy cleaning destroys sentence meaning, which is critical for mental health analysis, that why we used LIGHT TEXT CLEANING 

def light_clean(text):    
    text = str(text).lower()  #Lowercasing
    text = re.sub(r"http\S+|www\S+", "", text) #Remove URLs
    text = re.sub(r"\s+", " ", text).strip()  #Normalize Spaces
    return text

train_data['text'] = train_data['text'].apply(light_clean)


# 3. REMOVE MISSING VALUES

train_data.dropna(inplace=True)

print("\nClass distribution BEFORE balancing:")
print(train_data['status'].value_counts())


# 4. BALANCE THE DATASET

TARGET_SAMPLES_PER_CLASS = 5000  

balanced_data = []

for label in train_data['status'].unique():
    class_subset = train_data[train_data['status'] == label]

    if len(class_subset) >= TARGET_SAMPLES_PER_CLASS:
        class_subset = class_subset.sample(
            TARGET_SAMPLES_PER_CLASS,   #downsample reducing the number of samples in the majority class
            random_state=42
        )
    else:
        class_subset = class_subset.sample(
            TARGET_SAMPLES_PER_CLASS,
            replace=True,   #oversample increasing the number of samples in the minority class
            random_state=42
        )

    balanced_data.append(class_subset)

train_data = pd.concat(balanced_data).sample(frac=1, random_state=42)  #Merge all classes into one dataset

print("\nClass distribution AFTER balancing:")
print(train_data['status'].value_counts())

print("\nSample of balanced dataset:")
print(train_data.head(10))


# 5. LABEL ENCODING

label_mapping = {
    'Normal': 0,
    'Depression': 1,  # covert labels into numbers
    'Anxiety': 2,
    'Suicidal': 3
}

train_data['label'] = train_data['status'].map(label_mapping)


# 6. TRAIN / VALIDATION SPLIT

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_data['text'],
    train_data['label'],
    test_size=0.2,
    stratify=train_data['label'],
    random_state=42
)

print("\nTrain size:", len(train_texts))
print("Validation size:", len(val_texts))


# 7. TOKENIZATION (DISTILBERT) 
# converting raw text into numbers that a model like DistilBERT can understand.

tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(texts):
    return tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

X_train = tokenize(train_texts)
X_val = tokenize(val_texts)

y_train = torch.tensor(train_labels.tolist())
y_val = torch.tensor(val_labels.tolist())

print("\nTokenization completed")
print("Train input shape:", X_train['input_ids'].shape)
print("Validation input shape:", X_val['input_ids'].shape)


# 8. SAVE PREPROCESSED DATA

torch.save(
    {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val
    },
    "preprocessed_data_balanced.pt"
)






