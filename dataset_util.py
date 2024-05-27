def preprocess_function(examples,tokenizer):
    # Tokenize the texts
    inputs = tokenizer(
        examples["text"],
    )

    new_input = {
        "labels": examples["token_label_ids"],
    }
    inputs.update(new_input)
    return inputs


from transformers import AutoTokenizer


# def recreate_annotations_and_labels_with_longformer(text, original_tokens, original_labels):
#     tokenizer = AutoTokenizer.from_pretrained("hyperonym/xlm-roberta-longformer-base-16384")

#     # Tokenize the text and get offsets
#     encoded_input = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
#     new_tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'])
#     offsets = encoded_input['offset_mapping']

#     # Initialize the list for new labels with a default label (e.g., 'unknown')
#     new_labels = ['unknown'] * len(new_tokens)  # Default label for all tokens
#     label_texts = ['human', 'NLTK_synonym_replacement', 'chatgpt', 'summarized']

#     # Modify here to correctly handle spans
#     search_start = 0
#     for original_token, original_label in zip(original_tokens, original_labels):
#         start_position = text.find(original_token, search_start)  # Start searching from the last found position
#         if start_position == -1:
#             search_start = 0  # Reset search start if token not found (should not happen in well-formed data)
#             continue
#         end_position = start_position + len(original_token)
#         search_start = end_position  # Update search_start to the end of the current token

#         for i, (start, end) in enumerate(offsets):
#             if start >= start_position and end <= end_position:
#                 # Assign label to the new token
#                 new_labels[i] = label_texts[original_label]

#     # Group consecutive tokens with the same label
#     grouped_annotations = []
#     current_label = new_labels[0]
#     start_token_number = 0
#     for i, label in enumerate(new_labels + ['']):  # Add a dummy label to trigger the last group processing
#         if label != current_label:
#             grouped_annotations.append([start_token_number, i - 1, current_label])
#             current_label = label
#             start_token_number = i

#     # In this scenario, keep all annotations including 'unknown'
#     grouped_annotations = [group for group in grouped_annotations]

#     return new_tokens, grouped_annotations


def recreate_annotations_and_labels(examples,tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    label = examples[f"token_label_ids"]
    word_ids = tokenized_inputs.word_ids(batch_index=0)  # Map tokens to their respective word.
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(label[word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

def predict_original_labels(new_labels, original_tokens, text,tokenizer):
    encoded_input = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoded_input['offset_mapping']

    predict_labels = []  # Initialize the predicted labels list

    search_start = 0
    for original_token in original_tokens:
        start_position = text.find(original_token, search_start)
        if start_position == -1:
            # If the token cannot be found, it might be a tokenization error
            predict_labels.append(-1)  # Use -1 to denote an unknown label
            continue
        end_position = start_position + len(original_token)
        search_start = end_position  # Update the search start for the next token

        # Find the new tokens corresponding to the original token and their labels
        token_labels = []
        for i, (start, end) in enumerate(offsets):
            if start >= start_position and end <= end_position:
                token_labels.append(new_labels[i])

        # Calculate the occurrence count for each label
        label_counts = {label: token_labels.count(label) for label in set(token_labels)}

        # Identify the label with the highest occurrence count
        max_count = max(label_counts.values(), default=0)
        top_labels = [label for label, count in label_counts.items() if count == max_count]

        if len(top_labels) > 1 and predict_labels:  # If there are two or more labels with the same highest count, and it's not the first token
            predict_labels.append(predict_labels[-1])  # Use the label of the previous token
        elif token_labels:
            predict_labels.append(top_labels[0])  # Use the label with the highest occurrence count
        else:
            predict_labels.append(-1)  # If no label is found, use -1

    return predict_labels

