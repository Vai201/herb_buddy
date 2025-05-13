from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle
import os

# Load model, tokenizer, and label encoder
model = BertForSequenceClassification.from_pretrained("herbal_model")
tokenizer = BertTokenizer.from_pretrained("herbal_model")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Prediction function
def predict(query, model, tokenizer, device):
    encoding = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)

    return prediction.item()

# Interactive prediction
def interactive_predict():
    while True:
        query = input("Enter your herbal query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        predicted_id = predict(query, model, tokenizer, device)
        predicted_label = label_encoder.inverse_transform([predicted_id])[0]
        print(f"Predicted Plant ID: {predicted_label}")

# Start
interactive_predict()
