import pandas as pd
from transformers import ElectraForTokenClassification, ElectraTokenizerFast
import torch
import sys
model = ElectraForTokenClassification.from_pretrained(
    "./model_checkpoint_competition"
)
re = {"index":[],"preds":[] }
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
test_data = pd.read_parquet("./test_data.parquet", engine="fastparquet").values
def percent(current, total):

    if current == total - 1:
        current = total
    bar_length = 20
    hashes = "#" * int(current / total * bar_length)
    spaces = " " * (bar_length - len(hashes))
    sys.stdout.write(
        "\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total))
    )
span = 300
model.to("cuda")
def output_re(tok_text,label_ids):
    tok_text.pop("token_type_ids")
    with torch.no_grad():
        logits = model(**tok_text).logits
    raw_preds = torch.argmax(logits, dim=2)[0].cpu().numpy().tolist()
    word_ids =  tok_text.word_ids(batch_index=0)
    previous_word_idx = None
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx != previous_word_idx and word_idx is not None and word_idx<len(raw_preds):  # Only label the first token of a given word.
            label_ids.append(raw_preds[word_idx])
        previous_word_idx = word_idx
    return label_ids

for j,data in enumerate(test_data):
    percent(j,20000)
    re["index"].append(data[0])
    label_ids = []

    for k in range(len(data[2])//span):
        try:
            text = data[2][k*span:(k+1)*span]
            tok_text = tokenizer(text, is_split_into_words=True,return_tensors="pt").to("cuda")
            if len(tok_text["input_ids"][0]) >= 512:
                for i in range(len(text)//128):
                    text = text[i*128:(i+1)*128]
                    tok_text = tokenizer(text, is_split_into_words=True,return_tensors="pt").to("cuda")
                    if len(tok_text["input_ids"][0]) >= 512:
                        for e in range(len(text)//64):
                            text = text[e*64:(e+1)*64]
                            tok_text = tokenizer(text, is_split_into_words=True,return_tensors="pt").to("cuda")
                            if len(tok_text["input_ids"][0]) >= 512:
                                for n in range(len(text)//32):
                                    text = text[n*32:(n+1)*32]
                                    tok_text = tokenizer(text, is_split_into_words=True,return_tensors="pt").to("cuda")
                                    label_ids = output_re(tok_text,label_ids)
                            else:
                                label_ids = output_re(tok_text,label_ids)
                    else:
                        label_ids = output_re(tok_text,label_ids)

            else:
                label_ids = output_re(tok_text,label_ids)
        except Exception:
            print(text)
    re["preds"].append(label_ids)
    print(j)
df = pd.DataFrame(data=re)
df.to_parquet( 'predictions.parquet',engine='fastparquet')
