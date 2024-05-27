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
    })
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
trainer = MultitaskTrainer(model=multitask_model)
trainer._load_from_checkpoint("./model_checkpoint/checkpoint-5000/")
import pdb;pdb.set_trace()

import main
