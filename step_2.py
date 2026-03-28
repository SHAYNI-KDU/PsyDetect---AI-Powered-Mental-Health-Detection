import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score


# 1. LOAD PREPROCESSED DATA

data = torch.load("preprocessed_data_balanced.pt", weights_only=False)

X_train = data["X_train"]
X_val = data["X_val"]
y_train = data["y_train"]
y_val = data["y_val"]


# 2. DEVICE (CPU / GPU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 3. DATA LOADERS
#Hyperparameters
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5

train_dataset = TensorDataset(
    X_train["input_ids"],
    X_train["attention_mask"],
    y_train
)

val_dataset = TensorDataset(
    X_val["input_ids"],
    X_val["attention_mask"],
    y_val
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# 4. LOAD MODEL & TOKENIZER

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

model.to(device)


# 5. OPTIMIZER (Updates model weights during training)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


# 6. TRAINING LOOP (SAVE BEST MODEL)

best_macro_f1 = 0.0
MODEL_PATH = "mental_health_distilbert_best"

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 40)

    # ---- TRAIN ----
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch

        input_ids = input_ids.to(device)  #Move Data to CPU
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()    #Reset Gradients

        outputs = model(                   #Forward Pass  
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss          #Backward Pass (Learning)
        loss.backward()  #computes gradients (how wrong the model is)
        optimizer.step()  #updates weights to reduce future loss

        total_loss += loss.item()

    print(f"Training loss: {total_loss / len(train_loader):.4f}")

    # ---- VALIDATION ----
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print("\nValidation Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Normal", "Depression", "Anxiety", "Suicidal"]
    ))
    print(f"Macro F1-score: {macro_f1:.4f}")

    # ---- SAVE BEST MODEL ----
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)
        print("✅ Best model saved!") 

print("\nTraining complete!")
print(f"Best Macro F1: {best_macro_f1:.4f}")
print(f"Saved at: {MODEL_PATH}")
 


 # macro_f1 Computes F1 for each class and Averages them equally



