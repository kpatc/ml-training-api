# test.py
import argparse
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score,confusion_matrix,classification_report
import json
import joblib
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--dataset_id', type=int)
parser.add_argument('--project_name', type=str)
parser.add_argument('--task', type=str, choices=['similarity', 'inference'], default='similarity')
args = parser.parse_args()

model_path = f"storage/model/{args.project_name}/{args.dataset_id}/model.pkl"
vectorizer_path = f"storage/model/{args.project_name}/{args.dataset_id}/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer not found.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# model_path = f"storage/model/{args.dataset_id}/model"
file_path = f"storage/datasets/{args.project_name}/{args.dataset_id}/test.csv"

# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"Test file for dataset {args.dataset_id} not found")
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"No trained model found at {model_path}")

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = BertForSequenceClassification.from_pretrained(model_path)
# model.eval()

df = pd.read_csv(args.data)
df.dropna(inplace=True)

# if args.task == "similarity":
#     df = df[['textA', 'textB', 'label']]
# elif args.task == "inference":
#     df = df[['textA', 'textB', 'label']]
#     label_map = {"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}
#     df['label'] = df['label'].map(label_map)

# inputs = tokenizer(list(df["textA"]), list(df["textB"]), return_tensors="pt", padding=True, truncation=True, max_length=128)
# with torch.no_grad():
#     outputs = model(**inputs)

# preds = torch.argmax(outputs.logits, dim=1).numpy()
# labels = df['label'].values

# accuracy = accuracy_score(labels, preds)
# precision = precision_score(labels, preds, average='macro')
# recall = recall_score(labels, preds, average='macro')

# print(json.dumps({
#     "accuracy": round(accuracy, 4),
#     "precision": round(precision, 4),
#     "recall": round(recall, 4),
# }))

df['text'] = df['textA'] + " " + df['textB']
X = vectorizer.transform(df['text'])
y = df['label']

y_pred = model.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='macro')
recall = recall_score(y, y_pred, average='macro')
cm = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred, output_dict=True)

print(json.dumps({
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "confusion_matrix": cm.tolist(),
    "classification_report": report
}))
