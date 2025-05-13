import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import numpy as np
import pickle

# Load the data
data = pd.read_csv('herbal_data.csv')

# Encode the labels
label_encoder = LabelEncoder()
data['id'] = label_encoder.fit_transform(data['id'])
print(data['id'].unique())  # Before encoding
if predicted_id >= len(label_encoder.classes_):
    print("Predicted ID is out of range!")
else:
    original_label = label_encoder.inverse_transform([predicted_id])[0]
    print(f"Predicted Label: {original_label}")

# Save after fitting
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class QueryDataset(Dataset):
    def __init__(self, queries, labels, tokenizer, max_len):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, index):
        query = str(self.queries[index])
        label = self.labels[index]

        encoding = self.tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'query_text': query,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets and dataloaders
train_dataset = QueryDataset(
    queries=train_data['user_query'].values,
    labels=train_data['id'].values,
    tokenizer=tokenizer,
    max_len=128
)

test_dataset = QueryDataset(
    queries=test_data['user_query'].values,
    labels=test_data['id'].values,
    tokenizer=tokenizer,
    max_len=128
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(data['id'].unique()))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss().to(device)

# Training and evaluation functions
def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model.train()
    losses, correct_predictions = [], 0
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        labels = d['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        loss = loss_fn(outputs.logits, labels)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()
    losses, correct_predictions = [], 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels = d['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            loss = loss_fn(outputs.logits, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# Training loop
EPOCHS = 3
for epoch in range(EPOCHS):
    train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, len(train_data))
    print(f'Epoch {epoch+1}/{EPOCHS} - Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')

# Evaluate
test_acc, test_loss = eval_model(model, test_loader, loss_fn, device, len(test_data))
print(f'Test Accuracy: {test_acc:.4f}')

# Save model, tokenizer, and label encoder
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_tokenizer")
with open("label_encoder.pkl", 'wb') as f:
    pickle.dump(label_encoder, f)

# Prediction function
# Assuming the training has already completed, and now we want user input for predictions

# Loop to take input and provide prediction
def interactive_predict(model, tokenizer, device, label_encoder):
    while True:
        user_input = input("Enter your query (or type 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the chatbot.")
            break

        # Get prediction from the model
        predicted_id = predict(user_input, model, tokenizer, device)

        # Decode the predicted label
        original_label = label_encoder.inverse_transform([predicted_id])[0]
        print(f"Predicted Label: {original_label}")

# Call this function after training the model
interactive_predict(model, tokenizer, device, label_encoder)

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

    model.eval()
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(output.logits, dim=1)

    return prediction.item()

# Test prediction
new_query = "Are there any known side effects of using pumpkin for digestive health?"
predicted_id = predict(new_query, model, tokenizer, device)
original_label = label_encoder.inverse_transform([predicted_id])[0]
print(f'Predicted ID: {original_label}')
