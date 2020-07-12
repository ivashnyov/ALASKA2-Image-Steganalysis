import os
import random
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import albumentations as A
from sklearn.metrics import f1_score, roc_curve, auc
from catalyst.dl import MetricCallback
from catalyst.core import Callback, CallbackOrder, State
from efficientnet_pytorch import EfficientNet
from collections import defaultdict

class DatasetAlaska(Dataset):
    def __init__(self, kinds, image_names, labels, data_path, transforms=None):
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.data_path = data_path
        self.transforms = transforms

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        kind = self.kinds[idx]
        label = self.labels[idx]

        image = cv2.imread(f'{self.data_path}/{kind}/{image_name}')

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        label = torch.tensor(label).long()

        output = {
            "image": image,
            "label": label
        }      

        return output  

    def __len__(self):
        return len(self.image_names)

def weighted_auc(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = None,
        activation: str = None
):
    """
    Args:
        outputs (torch.Tensor): A list of predicted elements
        targets (torch.Tensor):  A list of elements that are to be predicted
        activation (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
    Returns:
        float: f1 score
    """
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    targets = targets.detach().cpu().numpy()
    targets[targets > 1] = 1
    outputs = 1 - nn.functional.softmax(outputs, dim=1).data.cpu().numpy()[:,0]
    fpr, tpr, thresholds = roc_curve(targets, outputs, pos_label=1)
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    
    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if mask.sum() == 0:
            continue
        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min # normalize such that curve starts at y=0
        score = auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric
        
    return competition_metric / normalization


class WeightedAUC(Callback):
    """
    F1 score metric callback.
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "weighted_auc",
            activation: str = None
    ):
        """
        Args:
            input_key (str): input key to use for iou calculation
                specifies our ``y_true``.
            output_key (str): output key to use for iou calculation;
                specifies our ``y_pred``
            activation (str): An torch.nn activation applied to the outputs.
                Must be one of ['none', 'Sigmoid', 'Softmax2d']
        """

        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.prefix = prefix
        self.predictions = defaultdict(lambda: [])
        self.metric_fn = weighted_auc

    def on_epoch_start(self, state) -> None:
        self.accum = []

    def on_loader_start(self, state: State):
        self.predictions = defaultdict(lambda: [])

    def on_batch_end(self, state: State) -> None:
        targets = state.input[self.input_key]
        outputs = state.output[self.output_key]
        self.predictions[self.input_key].append(targets.detach().cpu())
        self.predictions[self.output_key].append(outputs.detach().cpu())
        metric = self.metric_fn(outputs, targets)
        state.batch_metrics[f"batch_{self.prefix}"] = metric

    def on_loader_end(self, state) -> None:
        self.predictions = {
            key: torch.cat(value, dim=0)
            for key, value in self.predictions.items()
        }
        targets = self.predictions[self.input_key]
        outputs = self.predictions[self.output_key]
        value = self.metric_fn(
            outputs, targets
        )
        state.loader_metrics[self.prefix] = value

class AlaskaModel(nn.Module):
    def __init__(self, backbone='efficientnet-b0', classes=4):
        super(AlaskaModel, self).__init__()
        self.model = EfficientNet.from_pretrained(backbone, num_classes=classes)
        
    def forward(self, x):
        label = self.model(x)
        return {
            "logit_label": label
        }