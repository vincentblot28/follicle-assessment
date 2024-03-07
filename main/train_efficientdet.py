import datetime as dt
import json
import logging
import os
from typing import Dict, List

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils.efficientdet.dataset import EfficientDetDataModule
from utils.efficientdet.efficientdet import EfficientDetModel


EFFICIENTDET_MODELS_DIR = ".../data/03_model_weights/efficientdet"
MODEL_NAME = "tf_efficientdet_d4"
PATCH_SIZE = 512
TRAIN_VAL_SLIDES_DIR = ".../data/02_model_input/data_split.json"
PATCHES_PATH = ".../data/02_model_input/patches_stride_train_val"
TRAINING_EXAMPLES_PATH = ".../data/02_model_input/train_val_annotations_negative.json"
PATCHES_PATH_STRIDE = ".../data/02_model_input/patches_stride_train_val"
TRAINING_EXAMPLES_PATH_STRIDE = ".../data/02_model_input/patches_annotation_stride_train_val.json"
WITH_STRIDE = False
MODEL_PARAMS = {
    "learning_rate": 0.001,
    "prediction_confidence_threshold": 0.3,
    "wbf_iou_threshold": 0.3,
    "model_architecture": MODEL_NAME,
    "num_workers": 4,
    "batch_size": 25,
    "max_epochs": 100,
    "lr_warmup_epoch": 1
}
DEVICE = "gpu"


def train_efficientdet(effdet_models_dir: str,
                       model_name: str,
                       patch_size: int,
                       training_slides: List,
                       validation_slides: List,
                       patches_path: str,
                       training_examples_annotated: List[Dict],
                       model_params: Dict,
                       device: str = "gpu") -> Dict:
    """Node to train an EfficientDet model.

    Parameters
    ----------
    effdet_models_dir : str
        The directory of the EfficientDet trainings.
    model_name : str
        The base name of the model to create the experiment directory.
    patch_size : int
        Size of patches which will be created.
    training_slides : List
        A list of all the slides used for training.
    validation_slides : List
        A list of all the slides used for validation.
    patches_path : str
        The path where the patches are stored.
    training_examples_annotated : List[Dict]
        A list with the annotated examples for the slides in training and validation.
    model_params : Dict
        A dict with all the model parameters.
    device : str
        The device used to run the training. Run on GPU if device="gpu" else it runs on CPU.

    Returns
    -------
    Dict
        An empty dict. Every logs and models' weights are saved by pytorch-lightning with the Trainer.
        Models are saved with a ModelCheckpoint callback and metrics are logged into summaries for
        Tensorboard with a TensorBoardLogger.
    """

    current_date = dt.datetime.now().strftime('%Y%m%d_%H%M')
    date_model_name = '{}_{}_{}_lr_{}_pred_thr_{}_wbf_thr_{}'.format(
        current_date,
        model_name,
        patch_size,
        model_params["learning_rate"],
        model_params["prediction_confidence_threshold"],
        model_params["wbf_iou_threshold"]
    )
    model_dir = os.path.join(effdet_models_dir, date_model_name)
    os.makedirs(model_dir, exist_ok=True)

    logging.info('(object_detection_rgb) : start EfficientDet training with examples in path : {}'.format(patches_path))

    dm = EfficientDetDataModule(annotations=training_examples_annotated,
                                patches_path=patches_path,
                                training_slides=training_slides,
                                validation_slides=validation_slides,
                                img_size=patch_size,
                                num_workers=model_params["num_workers"],
                                batch_size=model_params["batch_size"])

    model = EfficientDetModel(num_classes=3,
                              img_size=patch_size,
                              prediction_confidence_threshold=model_params["prediction_confidence_threshold"],
                              learning_rate=model_params["learning_rate"],
                              wbf_iou_threshold=model_params["wbf_iou_threshold"],
                              model_architecture=model_params["model_architecture"],
                              lr_warmup_epoch=model_params["lr_warmup_epoch"]
                              )
    gpu = [0] if device.lower() == "gpu" else None
    logger = TensorBoardLogger(model_dir, name="training_logs")
    callbacks = [
        ModelCheckpoint(dirpath=model_dir,
                        save_top_k=1,
                        monitor='coco_AP_all',
                        mode='max',
                        save_last=True,
                        filename='effdet-epoch{epoch:02d}-AP{coco_AP_all:.2f}',
                        auto_insert_metric_name=False),
        LearningRateMonitor()
    ]
    trainer = Trainer(gpus=gpu,
                      max_epochs=model_params["max_epochs"],
                      logger=logger,
                      callbacks=callbacks)
    trainer.fit(model, dm)

    return {}


if __name__ == "__main__":
    with open(TRAIN_VAL_SLIDES_DIR, "r") as f:
        slides = json.load(f)
    with open(TRAINING_EXAMPLES_PATH if not WITH_STRIDE else TRAINING_EXAMPLES_PATH_STRIDE, "r") as f:
        examples = json.load(f)
    train_efficientdet(
        effdet_models_dir=EFFICIENTDET_MODELS_DIR,
        model_name=MODEL_NAME,
        patch_size=PATCH_SIZE,
        training_slides=slides["train"],
        validation_slides=slides["val"],
        patches_path=PATCHES_PATH if not WITH_STRIDE else PATCHES_PATH_STRIDE,
        training_examples_annotated=examples,
        model_params=MODEL_PARAMS,
        device=DEVICE
    )
