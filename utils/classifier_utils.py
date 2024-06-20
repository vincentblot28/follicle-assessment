import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomApply, RandomRotation, ColorJitter

class FollicleDataset():
    def __init__(self,dataset_type, root_img_dir, labels=None,transform = None):
        self.dataset_type = dataset_type
        self.root_img_dir = root_img_dir
        self.filelist = os.listdir(root_img_dir)
        self.transform = transform
        self.labels = labels
    def __len__(self):
        return int(len(self.filelist))
    def __getitem__(self,index):
        imgpath = os.path.join(self.root_img_dir, self.filelist[index])
        img = Image.open(imgpath)
        label = self.labels[list(self.labels.keys())[index]]
        if self.transform is not None:
            img = self.transform(img)
        if self.dataset_type == "train":    
            return img,label
        else:
            return img 

# LightningModule that receives a PyTorch model as input
class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        # The inherited PyTorch module
        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=['model'])

        # Set up attributes for computing the accuracy
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.valid_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
    # Defining the forward method is only necessary 
    # if you want to use a Trainer's .predict() method (optional)
    def forward(self, x):
        return self.model(x)
        
    # A common forward step to compute the loss and labels
    # this is used for training, validation, and testing below
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features).squeeze()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, true_labels)
        predicted_labels = F.sigmoid(logits)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("train_loss", loss)
        
        # To account for Dropout behavior during evaluation
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
        self.train_acc.update(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()
        return loss  # this is passed to the optimzer for training

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log("valid_acc", self.valid_acc,
                 on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, labels, batch_size):
        super().__init__()
        self.data_path = data_path
        self.labels = labels
        self.batch_size = batch_size
        
    def prepare_data(self):

        self.train_transform = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomApply(torch.nn.ModuleList([RandomRotation(degrees=(90, 90))]), p=.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly adjust color
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return

    def setup(self, stage=None):
        train = FollicleDataset(
            dataset_type="train", root_img_dir=self.data_path,
            labels=self.labels, transform=self.train_transform
        )

        self.test = FollicleDataset(
            dataset_type="test", root_img_dir=self.data_path,
            labels=self.labels, transform=self.test_transform
        )
        n_train = int(len(train) * 0.8)
        n_valid = len(train) - n_train
        self.train, self.valid = random_split(train, lengths=[n_train, n_valid])

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train, 
                                  batch_size=self.batch_size, 
                                  drop_last=True,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(dataset=self.valid, 
                                  batch_size=self.batch_size, 
                                  drop_last=False,
                                  shuffle=False)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test, 
                                 batch_size=self.batch_size, 
                                 drop_last=False,
                                 shuffle=False)
        return test_loader