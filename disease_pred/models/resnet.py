import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.auroc import AUROC
from torchvision import models


class ResNet(pl.LightningModule):
    """ResNet model with different depths."""

    def __init__(
        self,
        lr: float,
        pretrained: bool = True,
        model: str = "50",
        img_size: int = 224,
        n_channels: int = 5,
        n_classes: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        assert model in ["18", "34", "50", "101", "152"]

        # init a pretrained resnet
        if model == "18":
            backbone = models.resnet18(self.hparams.pretrained)
        elif model == "34":
            backbone = models.resnet34(self.hparams.pretrained)
        elif model == "50":
            backbone = models.resnet50(self.hparams.pretrained)
        elif model == "101":
            backbone = models.resnet101(self.hparams.pretrained)
        elif model == "152":
            backbone = models.resnet152(self.hparams.pretrained)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # for param in backbone.parameters():
        #    param.requires_grad = False

        # initial convolutions for conversion of 6 to 3 channels
        self.preconv0 = nn.Conv2d(in_channels=self.hparams.n_channels, out_channels=3, kernel_size=3, stride=1)

        # use the pretrained model to classify KWS scale (10 classes)
        self.classifier = nn.Linear(num_filters, self.hparams.n_classes)
        self.accuracy = Accuracy(num_classes=n_classes, average="weighted")
        self.auroc = AUROC(num_classes=n_classes, average="weighted")

    def forward(self, x):
        x = F.relu(self.preconv0(x))
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.hparams.lr)
        sch = {
            "scheduler": ReduceLROnPlateau(opt, mode="min", factor=0.9, patience=500, verbose=True),
            "interval": "step",
            "monitor": "train_loss",
            "name": "lr",
        }
        return {"optimizer": opt, "lr_scheduler": sch}

    def loss_function(self, logits, labels):
        return nn.CrossEntropyLoss()(logits.view(-1, self.hparams.n_classes), labels)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.loss_function(logits, labels["labels"][0])

        logits = nn.Softmax(-1)(logits)  # pdf
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels["labels"][0])
        auroc = self.auroc(logits, labels["labels"][0])

        self.log("train_loss", loss, prog_bar=False, sync_dist=True)
        self.log("train_acc", acc, prog_bar=False, sync_dist=True)
        self.log("train_auroc", auroc, prog_bar=False, sync_dist=True)

        return {"loss": loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)

        loss = self.loss_function(logits, labels["labels"][0])

        logits = nn.Softmax(-1)(logits)  # pdf
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels["labels"][0])
        auroc = self.auroc(logits, labels["labels"][0])

        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        self.log("val_acc", acc, prog_bar=False, sync_dist=True)
        self.log("val_auroc", auroc, prog_bar=True, sync_dist=True)

        return {"val_loss": loss, "val_acc": acc, "val_auroc": auroc}


if __name__ == "__main__":

    pass
