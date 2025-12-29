import os
import pandas as pd
from datasets import Dataset
from dateutil.utils import today
from google.cloud import storage
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from SingleAgent.common import compute_metrics, uploadDataToBucket, terminateVM

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
# Can also try "bert-base-uncased" and see if it's better but I assume not as caps matter in phishing, i.e URGENT vs Urgent

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)


train_ds = train_ds.map(tokenize_function, batched=True, num_proc=4)
val_ds = val_ds.map(tokenize_function, batched=True, num_proc=4)
test_ds = test_ds.map(tokenize_function, batched=True, num_proc=4)  # num_proc=4 parallel processing

train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

training_args = TrainingArguments(
    output_dir="Results/BertBodyAndSubj/results_bert_base_cased_3_epochs",
    eval_strategy="epoch",  # evaluates after each epoch
    save_strategy="epoch",  # Saves checkpoint after each epoch
    learning_rate=2e-5,  # step size, 2e-5 is standard for BERT
    per_device_train_batch_size=10,  # samples per gpu, 8 is standard
    per_device_eval_batch_size=12,  # same again
    num_train_epochs=3,  # number of epochs, 3 is standard
    weight_decay=0.01,  # prevents overfitting by penalising large weights, is default at 0.01
    logging_dir="Logs/BertBodyAndSubj/logs_bert_base_cased_3_epochs",
    logging_steps=1000,  # How often it logs metrics, 50 is defualt but large dataset means I've put it higher
    seed=1,  # for reproducibility
    fp16=True,  # helps speed and memory
    report_to=["tensorboard"]  # disable reporting
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


def main():
    # Ensure directories exist before saving
    os.makedirs("Results/Saves/BertBodyAndSubj/model", exist_ok=True)
    os.makedirs("Results/Saves/BertBodyAndSubj/tokenizer", exist_ok=True)
    
    trainer.train()

    metrics = trainer.evaluate(test_ds)
    trainer.save_model("Results/Saves/BertBodyAndSubj/model")
    tokenizer.save_pretrained("Results/Saves/BertBodyAndSubj/tokenizer")
    trainer.save_metrics("test", metrics=metrics, combined=True)
    trainer.save_state()


if __name__ == "__main__":
    try:
        main()
        # Upload training results
        uploadDataToBucket("Logs/BertBodyAndSubj/logs_bert_base_cased_3_epochs")
        uploadDataToBucket("Results/Saves/BertBodyAndSubj")
        uploadDataToBucket("Results/BertBodyAndSubj/results_bert_base_cased_3_epochs")
        
        # Upload the script file for reference
        script_path = "SingleAgent/bert_body_and_subj.py"
        if os.path.exists(script_path):
            client = storage.Client()
            bucket = client.get_bucket("model-storage-data")
            folder_name = today().strftime("%Y-%m-%d") + "_test1"
            blob = bucket.blob(f"{folder_name}/{script_path}")
            blob.upload_from_filename(script_path)
    except Exception as e:
        print(f"Error during training or upload: {e}")
        # Try to upload what we have even if training failed
        try:
            uploadDataToBucket("Results/Saves/BertBodyAndSubj")
            uploadDataToBucket("Logs/BertBodyAndSubj/logs_bert_base_cased_3_epochs")
            uploadDataToBucket("Results/BertBodyAndSubj/results_bert_base_cased_3_epochs")
        except Exception as upload_error:
            print(f"Error during upload: {upload_error}")
    finally:
        terminateVM()
