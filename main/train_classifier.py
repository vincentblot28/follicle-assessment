import json
from datetime import datetime
import sys
sys.path.append("/home/ubuntu/follicle-assessment")

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import models

from utils.classifier_utils import DataModule, LightningModel

IMAGES_DIR = "/home/ubuntu/folcon/02_model_input_classif/yolo/images/"
LABELS_PATH_JSON = "/home/ubuntu/folcon/02_model_input_classif/yolo/labels_dict.json"
BATCH_SIZE = 32


def train_classifier(images_dir, labels_path, batch_size):
    with open(labels_path, "r") as f:
        labels = json.load(f)

    data_module = DataModule(data_path=images_dir, labels=labels, batch_size=batch_size)
    model = models.vgg16(pretrained=True)
    # change the last layer to output 1 class
    model.classifier[6] = nn.Linear(4096, 1)
    # Add sigmoid layer
    # vgg16.classifier.add_module("7", nn.Sigmoid())

    lightning_model = LightningModel(
        model, learning_rate=1e-3)

    callbacks = [ModelCheckpoint(
        save_top_k=1, mode='max', monitor="valid_acc")]  # save top 1 model 
    model_name = datetime.now().strftime("%Y-%m-%d-%H-%M")
    logger = TensorBoardLogger(save_dir="logs/", name=model_name)
    trainer = pl.Trainer(
        max_epochs=200,
        callbacks=callbacks,
        accelerator="cuda",  # Uses GPUs or TPUs if available
        devices=[0],  # Uses all available GPUs/TPUs if applicable
        logger=logger,
        log_every_n_steps=100)
    trainer.fit(model=lightning_model, datamodule=data_module)


if __name__ == "__main__":
    train_classifier(IMAGES_DIR, LABELS_PATH_JSON, BATCH_SIZE)
    