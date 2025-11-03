from pathlib import Path

from dateutil.utils import today
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from google.cloud import storage
from googleapiclient import discovery
from google.oauth2 import service_account
import pandas as pd
from transformers.trainer_pt_utils import save_metrics

train_df = pd.read_csv("Data/all_data_train.csv")
val_df = pd.read_csv("Data/all_data_val.csv")
test_df = pd.read_csv("Data/all_data_test.csv")

train_df["label"] = train_df["label"].astype(int)
val_df["label"] = val_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)

def combine_text(row):
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + " [SEP] " + body.strip()

train_df["text"] = train_df.apply(combine_text, axis=1)
val_df["text"] = val_df.apply(combine_text, axis=1)
test_df["text"] = test_df.apply(combine_text, axis=1)

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
#Can also try "bert-base-uncased" and see if it's better but I assume not as caps matter in phishing, i.e URGENT vs Urgent

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

train_ds = train_ds.map(tokenize_function, batched=True, num_proc=8)
val_ds = val_ds.map(tokenize_function, batched=True, num_proc=8)
test_ds = test_ds.map(tokenize_function, batched=True, num_proc=8) #num_proc=4 parallel processing

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="Results/results_bert_base_cased_3_epochs",
    eval_strategy="epoch", #evaluates after each epoch
    save_strategy="epoch", #Saves checkpoint after each epoch
    learning_rate=2e-5, #step size, 2e-5 is standard for BERT
    per_device_train_batch_size=10, #samples per gpu, 8 is standard
    per_device_eval_batch_size=12, #same again
    num_train_epochs=3, #number of epochs, 3 is standard
    weight_decay=0.01, #prevents overfitting by penalising large weights, is default at 0.01
    logging_dir="Logs/logs_bert_base_cased_3_epochs",
    logging_steps=1000, #How often it logs metrics, 50 is defualt but large dataset means I've put it higher
    seed=1, #for reproducibility
    fp16=True, #helps speed and memory
    report_to=["tensorboard"] #disable reporting
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


def main():
    trainer.train()

    metrics = trainer.evaluate(test_ds)
    trainer.save_model("Results/Saves/model")
    save_metrics("Results/Saves", metrics)
    trainer.save_state()
    tokenizer.save_pretrained("Results/Saves/tokenizer")

def uploadDataToBucket(path_str):
    client = storage.Client()
    bucket = client.get_bucket("model-storage-data")
    folder_name = today().strftime("%Y-%m-%d") + "_test1"
    base = Path(path_str)
    for p in base.rglob("*"):
        if p.is_file():
            # Object name mirrors relative path under the provided directory
            rel = p.relative_to(base).as_posix()  # forward slashes
            blob = bucket.blob(f"{folder_name}/{base.as_posix()}/{rel}")
            blob.upload_from_filename(str(p))

def terminateVM():
    service = discovery.build('compute', 'v1')

    project = 'final-year-project-477110'
    zone = 'us-west1-a'
    instance_name = 'single-agent-training'

    request = service.instances().stop(project=project, zone=zone, instance=instance_name)
    response = request.execute()
    print("Stop request submitted:", response)

if __name__ == "__main__":
    try:
        main()
        uploadDataToBucket("Logs/logs_bert_base_cased_3_epochs")
        uploadDataToBucket("Results/Saves")
    except Exception as e:
        print(e)
        uploadDataToBucket("Logs/logs_bert_base_cased_3_epochs")

    client = storage.Client()
    bucket = client.get_bucket("model-storage-data")
    folder_name = today().strftime("%Y-%m-%d") + "_test1"
    blob = bucket.blob(folder_name)
    blob.upload_from_filename("main.py")
    terminateVM()

