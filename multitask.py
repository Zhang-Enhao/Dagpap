import numpy as np
import torch
import torch.nn as nn
import transformers
import logging
import pandas as pd
from datasets import Dataset
from transformers import ElectraForTokenClassification, AutoConfig
logging.basicConfig(level=logging.INFO)


dataset_dict = {
    "human": Dataset.from_pandas(pd.read_parquet("./train_300split.parquet", engine="fastparquet")),
    "synonym": Dataset.from_pandas(pd.read_parquet("./train_300split.parquet", engine="fastparquet")),
    "gpt": Dataset.from_pandas(pd.read_parquet("./train_300split.parquet", engine="fastparquet")),
    "summarization":Dataset.from_pandas(pd.read_parquet("./train_300split.parquet", engine="fastparquet")),
}


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                num_labels=2,
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
                                                                                                                                                                                                    1,1           Top
        """
        return "electra"

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

"""As described above, the `MultitaskModel` class consists of only two components - the shared "encoder", a dictionary to the individual task models. Now, we can simply create the corresponding task models by supplying the invidual model classes and model configs. We will use Transformers' AutoModels to further automate the choice of model class given a model architecture (in our case, let's use `roberta-base`)."""

model_name = "google/electra-large-discriminator"
multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "human": ElectraForTokenClassification,
        "synonym":ElectraForTokenClassification,
        "gpt": ElectraForTokenClassification,
        "summarization":ElectraForTokenClassification
    },
    # model_config_dict={
    #     "human": AutoConfig.from_pretrained(model_name, num_labels=2),
    #     "synonym": AutoConfig.from_pretrained(model_name, num_labels=2),
    #     "gpt": AutoConfig.from_pretrained(model_name, num_labels=2),
    #     "summarization": AutoConfig.from_pretrained(model_name, num_labels=2),
    # },
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

max_length =  512

def fn_human(example):
    example["labels"] = [0 if i==0 else 1 for i in example["labels"]]
    return example
def fn_synonym(example):
    example["labels"] = [0 if i==1 else 1 for i in example["labels"]]
    return example
def fn_gpt(example):
    example["labels"] = [0 if i==2 else 1 for i in example["labels"]]
    return example
def fn_summarization(example):
    example["labels"] = [0 if i==3 else 1 for i in example["labels"]]
    return example


convert_func_dict = {
    "human": fn_human,
    "synonym": fn_synonym,
    "gpt": fn_gpt,
    "summarization":fn_summarization,
}

"""Now that we have defined the above functions, we can use `dataset.map` method available in the NLP library to apply the functions over our entire datasets. The NLP library that handles the mapping efficiently and caches the features."""

columns_dict = {
    "human": ['input_ids', 'attention_mask', 'labels'],
    "synonym": ['input_ids', 'attention_mask', 'labels'],
    "gpt": ['input_ids', 'attention_mask', 'labels'],
    "summarization": ['input_ids', 'attention_mask', 'labels'],
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    features_dict[task_name] = dataset.map(
        convert_func_dict[task_name],
    )
    features_dict[task_name].set_format(
        type="torch",
        columns=columns_dict[task_name],
    )

from torch.utils.data.dataloader import DataLoader
from transformers.data.data_collator import DataCollator, InputDataClass,DefaultDataCollator,DataCollatorForTokenClassification
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict



class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
               collate_fn=self.data_collator.__call__,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })

"""## Time to train!

Okay, we have done all the hard work, now it is time for it to pay off. We can now simply create our `MultitaskTrainer`, and start training!

(This takes about ~45 minutes for me on Colab, but it will depend on the GPU you are allocated.)
"""

train_dataset = {
    task_name: dataset
    for task_name, dataset in features_dict.items()
}
trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
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
    learning_rate=1e-4,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-6,
    weight_decay=0.01,
    warmup_steps=10000,
    lr_scheduler_type="inverse_sqrt",
    max_grad_norm=5.0,
    ),
    data_collator=DataCollatorForTokenClassification(tokenizer),
    train_dataset=train_dataset,
)
trainer._load_from_checkpoint("./model_checkpoint/checkpoint-5000/")
import pdb;pdb.set_trace()