import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import model
import dataset_util
import numpy as np
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    ElectraForTokenClassification,
    ElectraTokenizerFast,
)
from datasets import Dataset,load_dataset
from functools import partial
import evaluate
import pandas as pd

# model = model.create_model("hyperonym/xlm-roberta-longformer-base-16384")
model = ElectraForTokenClassification.from_pretrained(
    "./model_checkpoint", num_labels=4,ignore_mismatched_sizes=True

)
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
# Training
df = pd.read_parquet("./train_300split.parquet", engine="fastparquet")
org_dataset = Dataset.from_pandas(df)
train_dataset = org_dataset.shuffle(seed=42)
#### No need pre_fn now. Already finished in data_split.
# pre_fn = partial(dataset_util.preprocess_function, tokenizer=tokenizer)
# train_dataset = train_dataset.map(pre_fn)["train"].shuffle(
#     seed=42
# )
# Eval
df = pd.read_parquet("./eval_300split.parquet", engine="fastparquet")
org_dataset = Dataset.from_pandas(df)
eval_dataset = org_dataset

seqeval = evaluate.load("seqeval")
label_dict = {0:"human", 1:"synonym", 2:"gpt",3:"summarized"}

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_dict[p] for (p, l) in zip(prediction, label) if l != -100 ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_dict[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


training_args = TrainingArguments(
    output_dir="./model_checkpoint_competition",
    per_device_train_batch_size=16,
    per_gpu_eval_batch_size=6,
    overwrite_output_dir=True,
    num_train_epochs=10,
    save_steps=5000,
    save_total_limit=5,
    report_to="wandb",
    remove_unused_columns=False,
    max_grad_norm=5.0,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    warmup_steps = 8000,
    lr_scheduler_type="inverse_sqrt",
    fp16=True,
    fp16_backend=True,
    fp16_full_eval=True,
    fp16_opt_level=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)
trainer.train()
trainer.save_model("./final_model")
