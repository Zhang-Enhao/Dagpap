import pandas as pd
from transformers import ElectraTokenizerFast,XLMRobertaTokenizerFast
import dataset_util
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
)
args = parser.parse_args()
data = pd.read_parquet(args.path, engine='fastparquet')
#new_data = {"input_ids":[],"labels":[], "attention_mask":[], "token_type_ids":[] }
#eval_data = {"input_ids":[],"labels":[], "attention_mask":[], "token_type_ids":[] }
new_data = {"input_ids":[],"labels":[], "attention_mask":[],  }
eval_data = {"input_ids":[],"labels":[], "attention_mask":[], }
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
#tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
span = 350
for j,d in enumerate(data.values):
    for k in range(len(d[-2])//span):
        new_labels = dataset_util.recreate_annotations_and_labels({"tokens":d[-2][k*span:(k+1)*span],f"token_label_ids":d[-1][k*span:(k+1)*span]},tokenizer)
        if j<= 4500:
            new_data["input_ids"].append([i for i in new_labels["input_ids"]])
            new_data["labels"].append(new_labels["labels"])
            new_data["attention_mask"].append(new_labels["attention_mask"])
            #new_data["token_type_ids"].append(new_labels["token_type_ids"])
        else:
            eval_data["input_ids"].append([i for i in new_labels["input_ids"]])
            eval_data["labels"].append(new_labels["labels"])
            eval_data["attention_mask"].append(new_labels["attention_mask"])
            #eval_data["token_type_ids"].append(new_labels["token_type_ids"])

df = pd.DataFrame(data=new_data)
df.to_parquet( 'train_300split.parquet',engine='fastparquet')
df = pd.DataFrame(data=eval_data)
df.to_parquet( 'eval_300split.parquet',engine='fastparquet')
