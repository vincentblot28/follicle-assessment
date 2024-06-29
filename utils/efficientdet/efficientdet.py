from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from effdet import DetBenchTrain, EfficientDet, get_efficientdet_config
from effdet.efficientdet import HeadNet
from ensemble_boxes import ensemble_boxes_wbf
from objdetecteval.metrics.coco_metrics import get_coco_stats
from pytorch_lightning import LightningModule

from utils.efficientdet.dataset import get_valid_transforms


def create_model(num_classes: int,
                 image_size: int,
                 architecture: str,
                 pretrained_backbone: bool) -> DetBenchTrain:
    """Create an EfficientDet model

    Parameters
    ----------
    num_classes : int
        The number of classes
    image_size : int
        The input image size
    architecture : str
        An architecture name to get the associated parameters
    pretrained_backbone : bool
        Use a pretrained backbone.

    Returns
    -------
    DetBenchTrain
        A model ready to be trained using the effdet package
    """
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=pretrained_backbone)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    return DetBenchTrain(net, config)


def run_wbf(predictions: List[Dict[str, Any]],
            image_size: int = 1024,
            iou_thr: float = 0.44,
            skip_box_thr: float = 0.0,
            weights=None) -> Tuple[List, List, List]:
    """Run the Weighted Boxes Fusion (WBF) algorithm

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        A list of dicts with the predictions. The dicts have keys: ["boxes", "scores", "classes"]
    image_size : int, optional
        The input image size, by default 1024
    iou_thr : float, optional
        The threshold on the IoU, by default 0.44
    skip_box_thr : float, optional
        The threshold to skip a box, by default 0.0
    weights : _type_, optional
        weights, by default None

    Returns
    -------
    Tuple[List, List, List]
        Three lists with respectively, the final bboxes, the confidence scores and the class labels.
    """
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]
        boxes_w, scores_w, labels_w = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes_w = boxes_w * (image_size - 1)
        bboxes.append(boxes_w.tolist())
        confidences.append(scores_w.tolist())
        class_labels.append(labels_w.tolist())

    return bboxes, confidences, class_labels


class EfficientDetModel(LightningModule):
    """This model is a pytorch-lightning module with all the material for training and prediction.

    Attributes
    ----------
    prediction_confidence_threshold : float
        The confidence threshold for which a prediction should be kept.
    lr : float
        The learning rate
    wbf_iou_threshold : float
        The IoU threshold used to match bboxes in the Weighted Box Fusion

    Methods
    ----------
    forward(images, targets)
        Do a forward pass on the images
    configure_optimizers()
        Get the configured optimizer for the training
    training_step(batch, batch_idx)
        Training set for a given batch
    validation_step(batch, batch_idx)
        Validation step for a given batch
    predict(images: List)
        Run a complete inference on the given images
    post_process_detections(detections)
        Do all the post processing needed on the detection outputs
    aggregate_prediction_outputs(outputs)
        Aggregate the outputs of the EfficientDet model into a formated output
    validation_epoch_end(outputs)
        Complete validation on all the validation set at the end of epochs
    _run_inference(images_tensor, image_sizes)
        Run the inference on the input tensors
    _create_dummy_inference_targets(num_images)
        Creates dummy targets for inference when no annotations are given
    _postprocess_single_prediction_detections(detections)
        Do the post processing on a unique detection
    __rescale_bboxes(predicted_bboxes, image_sizes)
        Rescale the bboxes to the image size
    """

    def __init__(self,
                 num_classes: int,
                 img_size: int,
                 prediction_confidence_threshold: float,
                 learning_rate: float,
                 wbf_iou_threshold: float,
                 model_architecture: str,
                 lr_warmup_epoch: int,
                 pretrained_backbone: bool = True):
        """EfficientDetModel constructor

        Parameters
        ----------
        num_classes : int
            Number of classes
        img_size : int
            The input image size
        prediction_confidence_threshold : float
            The confidence threshold for which a prediction should be kept.
        learning_rate : float
            The learning rate
        wbf_iou_threshold : float
            The IoU threshold used to match bboxes in the Weighted Box Fusion
        model_architecture : str
            The wanted backbone
        pretrained_backbone : bool, optional
            If you want to load the pretrained weights of the chosen backbone
            default is True, should be set to False when loading an already
            trained model.
        """
        super().__init__()
        self.img_size = img_size
        self.model = create_model(
            num_classes,
            img_size,
            model_architecture,
            pretrained_backbone
        )
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = get_valid_transforms()
        self.lr_warmup_epoch = lr_warmup_epoch

    def forward(self, images: torch.Tensor, targets: Dict) -> Dict[str, torch.Tensor]:
        """Forward pass

        Parameters
        ----------
        images : torch.Tensor
            Input images
        targets : Dict
            Targets

        Returns
        -------
        Dict[str, torch.Tensor]
            The model's output
        """
        return self.model(images, targets)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure a Adam optimizer for training

        Returns
        -------
         Dict[str, Any]
            A torch Adam optimizer and a learning rate scheduler
        """

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)

        sche = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=5,
                verbose=True, threshold=0.02, threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-08,
        )
        scheduler = {
            "scheduler": sche,
            "monitor": "valid_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Perform a training step

        Parameters
        ----------
        batch : Tuple
            The batch of input features
        batch_idx : int
            The batch index.

        Returns
        -------
        torch.Tensor
            The loss value for the step
        """
        images, annotations, _, _ = batch

        losses = self.model(images, annotations)

        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        return losses['loss']

    @torch.no_grad()
    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """Perform a validation step

        Parameters
        ----------
        batch : Tuple
            The batch of input features
        batch_idx : int
            The batch index.

        Returns
        -------
        Dict[str, Any]
            The loss value for the step and the batch predictions
        """
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)

        detections = outputs["detections"]

        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }

        logging_losses = {
            "class_loss": outputs["class_loss"].detach(),
            "box_loss": outputs["box_loss"].detach(),
        }

        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log(
            "valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True
        )
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)

        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}

    def predict(self, images: List) -> Tuple:
        """For making predictions from images

        Parameters
        ----------
        images: List
            A list of numpy images

        Returns
        -------
        Tuple
            A tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences
        """
        image_sizes = [(image.shape[1], image.shape[0]) for image in images]

        # resize if needed
        for i in range(len(images)):
            if self.img_size != images[i].shape[0]:
                images[i] = cv2.resize(images[i], (self.img_size,) * 2)
        images_tensor = torch.stack(
            [
                self.inference_tfms(
                    image=np.array(image, dtype=np.float32),
                    # labels=np.ones(1),
                    # bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]) -> Tuple:
        """Run inference

        Parameters
        ----------
        images_tensor : torch.Tensor
            Some images as pytorch Tensors
        image_sizes : List[Tuple[int, int]]
            The size of the input images

        Returns
        -------
        Tuple
            A tuple with the scaled_bboxes, predicted_class_labels, predicted_class_confidences, and the detections
        """
        dummy_targets = self._create_dummy_inference_targets(
            num_images=images_tensor.shape[0]
        )
        with torch.no_grad():
            detections = self.model(images_tensor.to(self.device), dummy_targets)["detections"]

        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        ordered_boxes = []
        for bboxes in scaled_bboxes:
            if len(bboxes) > 0:
                tmp_bboxes = np.array(bboxes)
                tmp_bboxes[:, [0, 1, 2, 3]] = tmp_bboxes[
                    :, [1, 0, 3, 2]
                ]  # Return axis to get [x_min, y_min, x_max, y_max]
                ordered_boxes.append((tmp_bboxes).tolist())

        return ordered_boxes, predicted_class_labels, predicted_class_confidences, detections

    def _create_dummy_inference_targets(self, num_images: int) -> Dict:
        """Create dummy targets for inference when no annotations are given

        Parameters
        ----------
        num_images : int
            Number of images which is the number of targets to be created

        Returns
        -------
        Dict
            The dummy targets as a Dict with keys: "bbox", "cls", "img_size" and "img_scale".
        """
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets

    def post_process_detections(self, detections: torch.Tensor) -> Tuple:
        """Perform the post processing on the detections

        Parameters
        ----------
        detections : torch.Tensor
            The outputs of the model

        Returns
        -------
        Tuple
            A tuple with the predicted_bboxes, predicted_class_confidences and predicted_class_labels.
        """
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(self._postprocess_single_prediction_detections(detections[i]))

        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels
        ) = run_wbf(predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold)

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections: torch.Tensor) -> Dict[str, Any]:
        """Perform the post processing on a single detection.

        Parameters
        ----------
        detections : torch.Tensor
            A single output of the model.

        Returns
        -------
        Dict[str, Any]
            A dict with keys: "boxes", "scores" and "classes".
        """
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes: List, image_sizes: List) -> List:
        """Rescale the bboxes to the image size.

        Parameters
        ----------
        predicted_bboxes : List
            The final predicted bboxes
        image_sizes : List
            The size on the images.

        Returns
        -------
        List
            A list with all the bboxes rescaled to the size of the image.
        """
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (
                        np.array(bboxes) * [
                            im_w / self.img_size,
                            im_h / self.img_size,
                            im_w / self.img_size,
                            im_h / self.img_size,
                        ]
                    ).tolist()
                )
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes

    def aggregate_prediction_outputs(self, outputs: List[Dict[str, Any]]) -> Tuple:
        """Aggregate the predictions in a unique output.

        Parameters
        ----------
        outputs : List[Dict[str, Any]]
            The efficientdet outputs

        Returns
        -------
        Tuple
            A tuple with the predicted_class_labels, image_ids, predicted_bboxes, predicted_class_confidences
            and the targets.
        """
        detections = torch.cat(
            [output["batch_predictions"]["predictions"] for output in outputs]
        )

        image_ids = []
        targets = []
        for output in outputs:
            batch_predictions = output["batch_predictions"]
            image_ids.extend(batch_predictions["image_ids"])
            targets.extend(batch_predictions["targets"])

        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        return (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets
        )

    def validation_epoch_end(self, outputs):
        """Compute and log training loss and accuracy at the epoch level.

        Compute the stats on the predictions with the coco API.

        Parameters
        ----------
        outputs : List[Dict[str, Any]]
            The outputs on which to perform the validation.

        Returns
        -------
        Dict
            The validation stats as a dict where the keys are metric names and values are the values of the metrics.
        """

        results = {}

        validation_loss_mean = torch.stack(
            [output["loss"] for output in outputs]
        ).mean()
        results["val_loss"] = validation_loss_mean

        (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets,
        ) = self.aggregate_prediction_outputs(outputs)

        truth_image_ids = [target["image_id"].detach().item() for target in targets]
        truth_boxes = [
            target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
        ]  # convert to xyxy for evaluation
        truth_labels = [target["labels"].detach().tolist() for target in targets]

        stats = get_coco_stats(
            prediction_image_ids=image_ids,
            predicted_class_confidences=predicted_class_confidences,
            predicted_bboxes=predicted_bboxes,
            predicted_class_labels=predicted_class_labels,
            target_image_ids=truth_image_ids,
            target_bboxes=truth_boxes,
            target_class_labels=truth_labels,
        )['All']
        results.update(stats)

        for k, v in results.items():
            if v < 0.0:
                results[k] = 0.0

        self.log('coco_AP_all', results["AP_all"])
        self.log('coco_AP_all_IOU_0_50', results["AP_all_IOU_0_50"])
        self.log('coco_AP_all_IOU_0_75', results["AP_all_IOU_0_75"])
        self.log('coco_AP_small', results["AP_small"])
        self.log('coco_AP_medium', results["AP_medium"])
        self.log('coco_AP_large', results["AP_large"])
        self.log('coco_AR_all_dets_1', results["AR_all_dets_1"])
        self.log('coco_AR_all_dets_10', results["AR_all_dets_10"])
        self.log('coco_AR_all', results["AR_all"])
        self.log('coco_AR_small', results["AR_small"])
        self.log('coco_AR_medium', results["AR_medium"])
        self.log('coco_AR_large', results["AR_large"])

        return results
