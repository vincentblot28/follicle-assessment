import glob
import os
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.core.composition import Compose
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def get_examples_by_slides(annotations: List[Dict[str, Any]],
                           slides: List[str],
                           exclude_negative_samples: bool = True) -> List[Dict[str, Any]]:
    """Get the examples from an annotation list based on the slides ids given in slides.

    It returns a subset of annotations with only the examples associated to a slide specified in slides.

    Parameters
    ----------
    annotations : List[Dict[str, Any]]
        A list with all the patches examples.
    slides : List[str]
        A list with the wanted slides
    exclude_negative_samples : bool, optional
        If true, it removes all the examples without annotations, by default True

    Returns
    -------
    List[Dict[str, Any]]
        A sub-list of annotations where only the examples of the specified slides are kept.
    """
    result = []
    for annot in annotations:
        if annot['slide'] in slides:
            if exclude_negative_samples:
                if len(annot['annotation']) > 0:
                    result.append(annot)
            else:
                result.append(annot)
    return result


def get_train_transforms() -> Compose:
    """Transformations to apply to the
    training images.

    Returns
    -------
    Compose
        Transformation of the train image.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.3),
            A.Blur(blur_limit=15, p=0.3),
            A.Rotate(limit=5, p=0.3),
            A.RandomRotate90(p=0.5),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ToTensorV2(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


def get_valid_transforms(with_ground_truth: bool = True) -> Compose:
    """Transformations to apply to the
    validation images.

    Attributes
    ----------
    with_ground_truth: bool
        If the examples have an associated ground truth bounding boxes
        or not, default at True.

    Returns
    -------
    Compose
        Transformation of the val image.
    """
    if with_ground_truth:
        transform = A.Compose(
            [
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(p=1),
            ],
            p=1.0,
            bbox_params=A.BboxParams(
                format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
            ),
        )
    else:
        transform = A.Compose(
            [
                A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ToTensorV2(p=1),
            ],
            p=1.0
        )
    return transform


class EfficientDetDataset(Dataset):
    """This class is a torch Dataset which outputs examples for the EfficientDet Model.

    Attributes
    ----------
    examples_path : str
        The path to the examples given in examples.
    examples : List
        The list of examples yield by the dataset.
    img_size : int
        The desired output size of the generated images.
    transforms : Compose
        The transformations to be applied to the generated items.

    Methods
    ----------
    _get_image_name_by_idx(index: int)
        Get the full image name by an index.
    _get_image_and_labels_by_idx(index: int)
        Load the image and associated annotations with an index.
    __getitem__(index: int)
        Iterator definition which loads an image and its associated annotations and format the result.
    __len__()
        Returns the len of the dataset which correspond to the number of examples.
    """

    def __init__(self,
                 examples_path: str,
                 examples: List[Dict],
                 img_size: int = 1024,
                 transforms: Compose = get_valid_transforms()):
        """EfficientDetDataset constructor

        Parameters
        ----------
        examples_path : str
            The path to the examples given in examples.
        examples : List[Dict]
            The list of examples yield by the dataset.
        img_size : int
            The desired output size of the generated images, by default 1024 (original size of a patch).
        transforms : Compose, optional
            The transformations to be applied to the generated items, by default get_valid_transforms().
        """
        self.examples_path = examples_path
        self.examples = examples
        self.img_size = img_size
        self.transforms = transforms

    def _get_image_name_by_idx(self, index: int) -> str:
        """Get the full image name by an index.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
        str
            The full image name
        """
        return self.examples[index]['full_name']

    def _get_image_and_labels_by_idx(self, index: int) -> Tuple:
        """Load the image and associated annotations with an index.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
        Tuple
            A tuple with the image, the bboxes with the pascal format, the class labels and the image id.

        Raises
        ------
        OSError
            If multiple files match the image name for an example.
        ValueError
            If the image file of the example does not end with 'png' nor 'npy'.
        """
        bb_annot = self.examples[index]

        search_image_filename = glob.glob(os.path.join(self.examples_path,
                                                       bb_annot["filepath"] + '.*'))
        if len(search_image_filename) == 0:
            raise OSError('No file match: ', search_image_filename)
        if len(search_image_filename) > 1:
            raise OSError('Multiple files match the example name in this directory: ', search_image_filename)
        image_filename = search_image_filename[0]

        if image_filename.endswith('.png'):
            image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image_filename.endswith('.npy'):
            image = np.load(image_filename)
        else:
            raise ValueError(f"The given image filename ({image_filename}) should end with either 'png' or 'npy'.")

        if len(bb_annot['annotation']) == 0:  # If no bbox in input image
            pascal_bboxes = np.array([], dtype=np.float32)
            class_labels = np.array([], dtype=np.float32)
        else:
            bboxes = np.array(
                [annot["bbox"] for annot in bb_annot["annotation"]],
                dtype=np.float32
            )
            # Format the bboxes to fit the expected pascal_voc format
            pascal_bboxes = np.stack([bboxes[:, 0],  # x_min
                                      bboxes[:, 1],  # y_min
                                      bboxes[:, 2],  # x_max
                                      bboxes[:, 3]],  # y_max
                                     axis=-1)
            class_labels = np.array(
                [annot["category"] + 1 for annot in bb_annot["annotation"]],
                dtype=np.float32
            )

        image = np.array(image, np.float32)

        pascal_bboxes *= self.img_size / image.shape[0]
        pascal_bboxes = np.clip(pascal_bboxes, 0, self.img_size)

        if self.img_size != image.shape[0]:
            image = cv2.resize(image, (self.img_size,) * 2)

        return image, pascal_bboxes, class_labels, bb_annot['id']

    def __getitem__(self, index: int) -> Tuple:
        """Iterator definition which loads an image and its associated annotations and format the result.

        Parameters
        ----------
        index : int
            Index

        Returns
        -------
        Tuple
            A tuple with the image, the correctly formated targets for EfficientDet and the image id.
        """
        (
            image,
            pascal_bboxes,
            class_labels,
            image_id,
        ) = self._get_image_and_labels_by_idx(index)

        # Apply transforms on data
        sample = {
            "image": image,
            "bboxes": pascal_bboxes,
            "labels": class_labels,
        }
        sample = self.transforms(**sample)

        image = sample["image"]
        bboxes = np.array(sample["bboxes"], np.float32)
        if len(bboxes) >= 1:
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]  # convert to [y_min, x_min, y_max, x_max]
            labels = sample["labels"]
        else:
            bboxes = np.array([[0, 0, 0, 0]])
            labels = np.array([0])

        _, new_h, new_w = image.shape
        target = {
            "bboxes": torch.as_tensor(bboxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.float32),
            "image_id": torch.tensor([image_id], dtype=torch.float32),
            "img_size": (new_h, new_w),
            "img_scale": torch.tensor([1.0], dtype=torch.float32),
        }

        return image, target, image_id

    def __len__(self) -> int:
        """Returns the length of the dataset.

        It corresponds to the number of examples.

        Returns
        -------
        int
            The dataset's length.
        """
        return len(self.examples)


class EfficientDetDataModule(LightningDataModule):
    """This class is a pytorch lightning DataModule.

    It interfaces the EfficientDetDataset.

    Attributes
    ----------
    patches_path : str
        The path to the patches.
    training_examples : List
        A list with all the training examples that should be find in the patches directory.
    validation_examples : List
        A list with all the validation examples that should be find in the patches directory.
    train_tfms : Compose
        The transforms to be applied to the training data.
    valid_tfms : Compose
        The transforms to be applied to the validation data.
    img_size : int
        The desired output image size.
    num_workers : int
        Number of workers for the data loader.
    batch_size : int
        Size of the example batches.

    Methods
    ----------
    train_dataset()
        Returns an EfficientDetDataset for training.
    train_dataloader()
        Returns a DataLoader by using the EfficientDetDataset generated from train_dataset().
    val_dataset()
        Returns an EfficientDetDataset for validation.
    val_dataloader()
        Returns a DataLoader by using the EfficientDetDataset generated from val_dataset().
    collate_fn()
        Defines how the examples yield by the EfficientDetDataset are merged to create batches.
    """

    def __init__(self,
                 annotations: List[Dict[str, Any]],
                 patches_path: str,
                 training_slides: List[str],
                 validation_slides: List[str],
                 train_transforms: Compose = get_train_transforms(),
                 valid_transforms: Compose = get_valid_transforms(),
                 img_size: int = 1024,
                 num_workers: int = 4,
                 batch_size: int = 8):
        """EfficientDetDataModule constructor

        Parameters
        ----------
        annotations : List[Dict[str, Any]]
            The list of all examples.
        patches_path : str
            The path to the patches.
        training_slides : List[str]
            A list with the slides that should be used for training.
        validation_slides : list[str]
            A list with the slides that should be used for validation.
        train_transforms : Compose, optional
            The transforms to be applied to the training data, by default get_train_transforms()
        valid_transforms : Compose, optional
            The transforms to be applied to the validation data, by default get_valid_transforms()
        img_size : int, optional
            The desired output image size, by default 1024
        num_workers : int, optional
            Number of workers for the data loader, by default 4
        batch_size : int, optional
            Size of the example batches, by default 8
        """
        self.patches_path = patches_path
        self.training_examples = get_examples_by_slides(annotations, training_slides, exclude_negative_samples=False)
        self.validation_examples = get_examples_by_slides(
            annotations, validation_slides, exclude_negative_samples=False
        )
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.img_size = img_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    # Training Dataset
    def train_dataset(self) -> EfficientDetDataset:
        """Returns an EfficientDetDataset for training.

        Returns
        -------
        EfficientDetDataset
            The training EfficientDetDataset
        """
        return EfficientDetDataset(examples_path=self.patches_path,
                                   examples=self.training_examples,
                                   img_size=self.img_size,
                                   transforms=self.train_tfms)

    def train_dataloader(self) -> DataLoader:
        """Returns a DataLoader by using the EfficientDetDataset generated from train_dataset().

        Returns
        -------
        DataLoader
            The DataLoader for training.
        """
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader

    # Validation Dataset
    def val_dataset(self) -> EfficientDetDataset:
        """Returns an EfficientDetDataset for validation.

        Returns
        -------
        EfficientDetDataset
            The validation EfficientDetDataset
        """
        return EfficientDetDataset(examples_path=self.patches_path,
                                   examples=self.validation_examples,
                                   img_size=self.img_size,
                                   transforms=self.valid_tfms)

    def val_dataloader(self) -> DataLoader:
        """Returns a DataLoader by using the EfficientDetDataset generated from val_dataset().

        Returns
        -------
        DataLoader
            The DataLoader for validation.
        """
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return valid_loader

    @staticmethod
    def collate_fn(batch: List) -> Tuple:
        """Defines how the examples yield by the EfficientDetDataset are merged to create batches.

        Parameters
        ----------
        batch : List
            The batch with stacked elements.

        Returns
        -------
        Tuple
            The merge elements in a single Tuple with: a single tensor with the images, a annotations dict
            with single tensors for every attribute, the original targets and images_ids.
        """
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids
