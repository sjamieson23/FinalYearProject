import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers.trainer_pt_utils import save_metrics

from SingleAgent.common import compute_metrics

# test_df = pd.read_csv("Data/all_data_test.csv")
test_df = pd.read_csv("Data/spear_test.csv")
test_df["label"] = test_df["label"].astype(int)


def combine_text(row):
    subj = str(row["subject"]) if not pd.isna(row["subject"]) else ""
    body = str(row["body"]) if not pd.isna(row["body"]) else ""
    return subj.strip() + " [SEP] " + body.strip()


test_df["text"] = test_df.apply(combine_text, axis=1)

test_ds = Dataset.from_pandas(test_df)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)


test_ds = test_ds.map(tokenize_function, batched=True, num_proc=4)  # num_proc=4 parallel processing
test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("PreTrained/BertBodyAndSubj", num_labels=2)

training_args = TrainingArguments(
    output_dir="Results/Eval_Pretrained/BertBodyAndSubj",
    per_device_eval_batch_size=12,
    report_to=["none"]
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


def fromPretrained():
    metrics = trainer.evaluate(test_ds)
    print("Evaluation metrics:", metrics)

    save_metrics("Results/Eval_Pretrained/BertBodyAndSubj", metrics=metrics, combined=True)


if __name__ == "__main__":
    fromPretrained()
