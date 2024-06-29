from datetime import datetime
import json
import sys
sys.path.append("/home/ubuntu/follicle-assessment")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torchvision import datasets, models
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip, RandomVerticalFlip, RandomApply, RandomRotation, ColorJitter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils.classifier_utils import FollicleDataset
IMAGE_PATH = "/home/ubuntu/ssd/folcon/02_model_input_classif/yolo/images2/"
LABELS_PATH_JSON = "/home/ubuntu/ssd/folcon/02_model_input_classif/yolo/labels.json"

# Load labels
with open(LABELS_PATH_JSON) as f:
    labels = json.load(f)
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = Compose([
            # RandomHorizontalFlip(p=0.5),
            # RandomVerticalFlip(p=0.5),
            # RandomApply(torch.nn.ModuleList([RandomRotation(degrees=(90, 90))]), p=.5),
            # ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),  # Randomly adjust color
            # RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
test_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

all_dataset = FollicleDataset(
            dataset_type="train", root_img_dir=IMAGE_PATH,
            labels=labels, transform=train_transform
        )
# Load data
n_train = int(len(all_dataset) * 0.8)
n_valid = len(all_dataset) - n_train
train_dataset, valid_dataset = random_split(all_dataset, lengths=[n_train, n_valid])
valid_dataset.transform = test_transform


image_datasets = {
    'train': train_dataset,
    'val': valid_dataset
}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
lr = 1e-6
wd = 0.01

#  Load the pre-trained VGG16 model
model = models.vgg16(pretrained=True)


# Modify the classifier part of the model for binary classification
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, 2)

model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

# Training and evaluation function
# Training and evaluation function
model_name = datetime.now().strftime("%Y-%m-%d-%H-%M")
model_name = "resnet152_" + f"{model_name}_lr={lr}_wd={wd}"
writer = SummaryWriter(f'runs/{model_name}')

def train_model(model, criterion, optimizer, num_epochs=25, log_interval=20):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Log the batch loss and accuracy every `log_interval` steps
                if phase == 'train' and global_step % log_interval == 0:
                    current_loss = running_loss / ((step + 1) * inputs.size(0))
                    current_acc = running_corrects.double() / ((step + 1) * inputs.size(0))
                    writer.add_scalar('Loss/train', current_loss, global_step)
                    writer.add_scalar('Accuracy/train', current_acc, global_step)
                    print(f'Step {global_step}: Train Loss: {current_loss:.4f}, Train Acc: {current_acc:.4f}')

                global_step += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Log the epoch loss and accuracy to TensorBoard
            if phase == 'val':
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print('Best val Acc: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train and evaluate the model
model = train_model(model, criterion, optimizer, num_epochs=25)

# Close the TensorBoard writer
writer.close()

# Save the model
torch.save(model.state_dict(), f'model_weights/vgg16.pth')