import os
import pandas as pd
from utils import *
import collections
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from catalyst.utils import set_global_seed
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl import CriterionCallback
from catalyst.dl.callbacks import EarlyStoppingCallback, MetricAggregationCallback
from catalyst.contrib.nn.optimizers import RAdam, Lookahead, Lamb
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = ArgumentParser("Alaska")

parser.add_argument("--data_folder", type=str, help="Path to folder with data", default="../input/alaska2-image-steganalysis/")
parser.add_argument("--image_height", type=int, help="Height of images to train", default=512)
parser.add_argument("--image_width", type=int, help="Width of images to train", default=512)
parser.add_argument("--batch_size", type=int, help="Batch size", default=36)
parser.add_argument("--num_workers", type=int, help="Number of workers", default=12)
parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=40)
parser.add_argument("--lr", type=int, help="Starting learning rate", default=3e-4)
parser.add_argument("--log_path", type=str, help="Path to logs", default="logs")
parser.add_argument("--fold", type=int, help="Fold to validate on", default=0)
parser.add_argument("--model", type=str, help="Model to train", default='efficientnet-b6')

args = parser.parse_args()

set_global_seed(42)

DATA_FOLDER = args.data_folder
FOLD = args.fold

dataset = pd.read_csv(f"{DATA_FOLDER}/data.csv")

transforms_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=args.image_height, width=args.image_width, p=1.0),
            A.Normalize(p=1.0),
        ], p=1.0)

transforms_val = A.Compose([
            A.Resize(height=args.image_height, width=args.image_width, p=1.0),
            A.Normalize(p=1.0),
        ], p=1.0)

train_dataset = DatasetAlaska(
    kinds=dataset[dataset['fold'] == FOLD].kind.values,
    image_names=dataset[dataset['fold'] == FOLD].image_name.values,
    labels=dataset[dataset['fold'] == FOLD].label.values,
    data_path=DATA_FOLDER,
    transforms=transforms_train
)
val_dataset = DatasetAlaska(
    kinds=dataset[dataset['fold'] != FOLD].kind.values,
    image_names=dataset[dataset['fold'] != FOLD].image_name.values,
    labels=dataset[dataset['fold'] != FOLD].label.values,
    data_path=DATA_FOLDER,
    transforms=transforms_val
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=False,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=False,
    shuffle=False
)

model = AlaskaModel(backbone=args.model, classes=4)
model.cuda()

loaders = collections.OrderedDict()
loaders["train"] = train_loader
loaders["valid"] = val_loader

runner = SupervisedRunner(input_key="image", output_key=None, input_target_key=None)

optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=0.001)

scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.75, patience=3, mode="max")

criterion = {
    'label_loss': nn.CrossEntropyLoss()
    }

callbacks = [
    CriterionCallback(
        input_key="label",
        output_key="logit_label",
        prefix="label_loss",
        criterion_key="label_loss",
        multiplier=1.0,
    ),
    MetricAggregationCallback(
        prefix="loss",
        metrics=[
            "label_loss",
        ],
    ),
    WeightedAUC(
        input_key="label",
        output_key="logit_label"
    )
]

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    callbacks=callbacks,
    loaders=loaders,
    logdir=args.log_path,
    scheduler=scheduler,
    fp16=True,
    num_epochs=args.num_epochs,
    verbose=True
)



