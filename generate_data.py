import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForWholeWordMask,  Trainer, TrainingArguments,AutoTokenizer,BertForMaskedLM, ElectraForTokenClassification,ElectraTokenizerFast, DataCollatorForTokenClassification
from datasets import load_dataset
from dataset_util import recreate_annotations_and_labels
import evaluate
import numpy as np
import collections
import numpy as np
import json

from transformers import default_data_collator
import pandas as pd
wwm_probability = 0.1

def percent(current, total):
    import sys

    if current == total - 1:
        current = total
    bar_length = 20
    hashes = "#" * int(current / total * bar_length)
    spaces = " " * (bar_length - len(hashes))
    sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total)))

def whole_word_masking_data_collator(feature, tokenizer):
    word_ids = feature["word_ids"]

    # Create a map between words and corresponding token indices
    mapping = collections.defaultdict(list)
    current_word_index = -1
    current_word = None
    for idx, word_id in enumerate(word_ids):
        if word_id is not None:
            if word_id != current_word:
                current_word = word_id
                current_word_index += 1
            mapping[current_word_index].append(idx)

    # Randomly mask words
    mask = np.random.binomial(1, wwm_probability, (len(mapping),))
    input_ids = feature["input_ids"][0]
    for word_id in np.where(mask)[0]:
        word_id = word_id.item()
        for idx in mapping[word_id]:
            input_ids[idx] = tokenizer.mask_token_id

    return torch.unsqueeze(input_ids,0)
# 初始化分词器
# tokenizer = ElectraTokenizer.from_pretrained("google/electra-small-generator")
sciBERT_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
sciModel = BertForMaskedLM.from_pretrained('allenai/scibert_scivocab_uncased')
# 加载和预处理数据集
dataset = load_dataset("text", data_files="pubmed-dataset/train.txt",streaming=True)
wholeWordMask = DataCollatorForWholeWordMask(sciBERT_tokenizer)
model = ElectraForTokenClassification.from_pretrained(
    "google/electra-large-discriminator", num_labels=2
)
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

def tokenize_and_encode(examples):
    inputs = sciBERT_tokenizer(examples,padding=True, truncation=True, max_length=512,return_tensors="pt")
    inputs["word_ids"] = inputs.word_ids(batch_index=0)
    inputs["labels"] = inputs["input_ids"]
    inputs["input_ids"] = whole_word_masking_data_collator(inputs,sciBERT_tokenizer)
    inputs.pop("word_ids")
    sciModel_prediction = sciBERT_tokenizer.decode(torch.argmax(sciModel(**inputs).logits[0][1:-1],-1))
    labels = []
    org_token = sciBERT_tokenizer.decode(sciBERT_tokenizer.encode(examples, add_special_tokens=False)).split(" ")
    predict_token = sciModel_prediction.split(" ")
    labels = np.zeros(len(predict_token),dtype=int)
    ranger_it = len(org_token) if len(org_token) < len(predict_token) else len(predict_token)
    for k in range(ranger_it):
        if org_token[k]== predict_token[k]:
            labels[k]=1
    new_labels = recreate_annotations_and_labels({"tokens":predict_token,"token_label_ids":labels},tokenizer)
    del inputs
    inputs = tokenizer(predict_token,padding=True, truncation=True, max_length=512,return_tensors="pt",is_split_into_words=True)
    inputs["labels"] = torch.unsqueeze(torch.Tensor(new_labels["labels"]),0)
    return inputs

new_data = {"input_ids":[],"labels":[], "attention_mask":[], "token_type_ids":[] }
n = 350
output_count = 0
for k, exmples in enumerate(dataset["train"]):
    output_count +=1
    percent(output_count,203037)
    input_data = " ".join(json.loads(exmples["text"])["article_text"])
    input_data = input_data.split(" ")
    for i in range(len(input_data)//n):
        inputs = tokenize_and_encode(" ".join(input_data[i*n: (i+1)*n]))
        new_data["input_ids"].append(inputs["input_ids"][0].numpy().tolist())
        new_data["labels"].append(inputs["labels"][0].numpy().tolist())
        new_data["attention_mask"].append(inputs["attention_mask"][0].numpy().tolist())
        new_data["token_type_ids"].append(inputs["token_type_ids"][0].numpy().tolist())
    if output_count > 1000:
        df = pd.DataFrame(data=new_data)
        df.to_parquet('./scientific_papers/' + str(k) + '_pretrain_data_pubmed-dataset.parquet',engine='fastparquet')
        new_data = {"input_ids":[],"labels":[], "attention_mask":[], "token_type_ids":[] }
        n = 350
        output_count = 0
