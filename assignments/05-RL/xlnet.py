from transformers import (
    XLNetForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    glue_tasks_num_labels,
)
import torch
from datasets import load_dataset, get_dataset_split_names, DatasetDict
import numpy as np
import evaluate


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluate.load("glue", "qnli").compute(
        predictions=predictions, references=labels
    )


def tokenize_function(examples):
    # print(examples["sentence"])
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)


# set device to GPU if available

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qnli = load_dataset(
    "glue", "qnli", split=["train[:1%]", "test[:1%]", "validation[:1%]"]
)
qnli_dict = {}
qnli_dict["train"] = qnli[0]
qnli_dict["test"] = qnli[1]
qnli_dict["validation"] = qnli[2]
qnli_dict = DatasetDict(qnli_dict)
# load the pre-trained XLNet model and tokenizer
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# set the GLUE task to be solved (e.g. 'mrpc', 'sst-2', 'qnli', etc.)
task = "qnli"
# load the training and validation datasets from GLUE
train_dataset = qnli_dict["train"]
valid_dataset = qnli_dict["validation"]
tokenized_datasets = qnli_dict.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42)
valid_dataset = tokenized_datasets["validation"].shuffle(seed=42)
# set the number of labels and the performance metric for evaluation
# num_labels = glue_tasks_num_labels[task]['num_labels']
num_labels = 2
# metric_name = glue_tasks_num_labels[task]['metric']
# metric = glue_tasks_num_labels[task]['metric_func']
# set the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=1000,
    save_total_limit=1,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir="./logs",
    # logging_steps=1000,
    # load_best_model_at_end=False,
    # metric_for_best_model=metric_name,
    # greater_is_better=True
    save_strategy="no",
)
# create the Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# train the model
trainer.train()
# evaluate the model
eval_results = trainer.evaluate()
print(eval_results)
