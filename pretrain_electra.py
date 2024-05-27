import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForWholeWordMask,  Trainer, TrainingArguments,AutoTokenizer,BertForMaskedLM, ElectraForTokenClassification,ElectraTokenizerFast, DataCollatorForTokenClassification
from datasets import load_dataset
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
import pandas as pd
import glob

file_list = glob.glob("./scientific_papers/*parquet")
main_dataframe = pd.read_parquet(file_list[0],engine="fastparquet")

for i in range(1,len(file_list)):
    df =  pd.read_parquet(file_list[i], engine="fastparquet")
    main_dataframe = pd.concat([main_dataframe, df], axis = 0)


#dataset = load_dataset("parquet",data_files="./scientific_papers/*parquet",streaming=True)
#train_data = Dataset.from_pandas(main_dataframe).remove_columns("__index_level_0__")
#test =  pd.read_parquet("./pretrain_test.parquet", engine="fastparquet")
#eval_data = Dataset.from_pandas(test)
model = ElectraForTokenClassification.from_pretrained(
    "google/electra-large-discriminator", num_labels=2
)
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
def to_binary(example):
    example["labels"] = [0 if i==0 or i==1 else 1 for i in example["labels"]]
    return example
# Training
df = pd.read_parquet("./train_300split.parquet", engine="fastparquet")
org_dataset = Dataset.from_pandas(df)
train_data = org_dataset.map(to_binary).shuffle(seed=42)
#### No need pre_fn now. Already finished in data_split.
# pre_fn = partial(dataset_util.preprocess_function, tokenizer=tokenizer)
# train_dataset = train_dataset.map(pre_fn)["train"].shuffle(
#     seed=42
# )
# Eval
df = pd.read_parquet("./eval_300split.parquet", engine="fastparquet")
org_dataset = Dataset.from_pandas(df)
eval_data = org_dataset.map(to_binary)


seqeval = evaluate.load("seqeval")
label_dict = {1:"fake", 0:"real"}

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
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)
training_args = TrainingArguments(
    output_dir="./model_checkpoint",
    per_device_train_batch_size=16,
    per_gpu_eval_batch_size=6,
    overwrite_output_dir=True,
    num_train_epochs=2,
    save_steps=5000,
    save_total_limit=5,
    report_to="wandb",
    remove_unused_columns=False,
    fp16=True,
    fp16_backend=True,
    fp16_full_eval=True,
    fp16_opt_level=True,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=10000,
    lr_scheduler_type="inverse_sqrt",
    max_grad_norm=5.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./model_checkpoint")

eval_result = trainer.evaluate()
print("Evaluation results:", eval_result)

