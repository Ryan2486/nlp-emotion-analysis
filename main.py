print("Loading a dataset from Hugging Face Datasets")
from datasets import load_dataset

emotions = load_dataset("emotion")
print(f"Number of samples in the emotion dataset: {len(emotions)}")
print(f'2 Example data for training :{emotions["train"][:2]}')

print("Loading a pre-trained tokenizer from Hugging Face Transformers")
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
print("Tokenizer loaded successfully!")

print("Tokenizing a sample text using the pre-trained tokenizer")
text = "Tokenizing text is a core task of NLP"
print("text:", text)
encoded_text = tokenizer(text)
print(encoded_text)
tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print(tokens)

print("Define a function to tokenize a batch of texts")


def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)


print("Tokenizing the first two samples from the training set")
print(tokenize(emotions["train"][:2]))

print("Tokenizing the entire training set")
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print("Tokenization complete!")
print("Checking the structure of the tokenized dataset :", emotions_encoded["train"].column_names)

print("Loading a pre-trained encoder/model from Hugging Face Transformers")
import torch
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
print("Encoder/model loaded successfully!")

print("Defining a function to extract hidden states from the model")


def extract_hidden_states(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
    return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}


print("Extracting hidden states from the tokenized dataset")
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)
print("Hidden states extracted successfully!")

print("Checking the structure of the dataset with hidden states:", emotions_hidden["train"].column_names)

print("Converting the hidden states to a NumPy array")
import numpy as np

X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])

print("Shape of the training and validation data:")
print(X_train.shape, X_valid.shape)

print("Shape of the training and validation labels:")
print(y_train.shape, y_valid.shape)

print("Embedding the training data using UMAP")
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler

X_scaled = MinMaxScaler().fit_transform(X_train)
mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = y_train
print("Shape of the UMAP embedding DataFrame:", df_emb.shape)

print("Visualizing the UMAP embedding of the training data")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotions["train"].features["label"].names
for i, (label, cmap) in enumerate(zip(labels, cmaps)):
    df_emb_sub = df_emb.query(f"label == {i}")
    axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
                   gridsize=20, linewidths=(0,))
    axes[i].set_title(label)
    axes[i].set_xticks([]), axes[i].set_yticks([])
plt.tight_layout()
plt.show()

print("Training a logistic regression classifier on the hidden states")
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)
print("Training complete! Accuracy on validation set:", lr_clf.score(X_valid, y_valid))

print("Evaluating the classifier on the validation set")
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


y_preds = lr_clf.predict(X_valid)
plot_confusion_matrix(y_preds, y_valid, labels)

print("Fine-tuning a pre-trained model for sequence classification")
from transformers import AutoModelForSequenceClassification

num_labels = 6
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))
print("Model for sequence classification loaded successfully!")

print("Defining a function to calculate metrics for evaluation")
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


print("Setting up the training arguments for fine-tuning")
from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"])
model_name = f"{model_ckpt}-finetuned-emotion"
training_args = TrainingArguments(output_dir=model_name, num_train_epochs=2, learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=True,
                                  log_level="error")

print(" Creating a Trainer instance for fine-tuning")
trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

print("Fine-tuning complete! Evaluating the model on the validation set")
preds_output = trainer.predict(emotions_encoded["validation"])
print("Evaluation results:", preds_output.metrics)

print("Plotting the confusion matrix for the fine-tuned model")
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)

