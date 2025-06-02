import argparse
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import json

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float)
parser.add_argument('--epochs', type=int)  # not used but kept for compatibility
parser.add_argument('--batch', type=int)   # same
parser.add_argument('--data', type=str)
parser.add_argument('--dataset_id', type=int)
parser.add_argument('--project_name', type=str)
parser.add_argument('--task', type=str, choices=['similarity', 'inference'], default='similarity')
args = parser.parse_args()

# --------- STEP 1: Load data ---------
print(json.dumps({"log": "Loading data..."}))
df = pd.read_csv(args.data)
df.dropna(inplace=True)

if args.task == "similarity":
    df = df[['textA', 'textB', 'label']]
    num_labels = 2
elif args.task == "inference":
    df = df[['textA', 'textB', 'label']]
    label_map = {"ENTAILMENT": 0, "CONTRADICTION": 1, "NEUTRAL": 2}
    df['label'] = df['label'].map(label_map)
    num_labels = 3

df['text'] = df['textA'] + " " + df['textB']

# --------- STEP 2: Train/Val split ---------
print(json.dumps({"log": "Splitting data into train/validation sets..."}))
X_train, X_val, y_train, y_val = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# --------- STEP 3: TF-IDF Vectorization ---------
print(json.dumps({"log": "Applying TF-IDF vectorization..."}))
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

# --------- STEP 4: Model Training ---------
print(json.dumps({"log": "Training Logistic Regression model..."}))
model = LogisticRegression(max_iter=1000, C=args.lr)
model.fit(X_train_vec, y_train)

# --------- STEP 5: Prediction ---------
print(json.dumps({"log": "Predicting on validation set..."}))
y_pred = model.predict(X_val_vec)

# --------- STEP 6: Metrics ---------
print(json.dumps({"log": "Computing evaluation metrics..."}))
acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='macro')
cm = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred, output_dict=True)

# --------- STEP 7: Saving Model ---------
print(json.dumps({"log": "Saving model and vectorizer..."}))
save_path = f"storage/model/{args.project_name}/{args.dataset_id}/"
os.makedirs(save_path, exist_ok=True)
joblib.dump(model, os.path.join(save_path, 'model.pkl'))
joblib.dump(vectorizer, os.path.join(save_path, 'vectorizer.pkl'))

# --------- STEP 8: Final Output ---------
print(json.dumps({
    "metrics": {
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
}))
