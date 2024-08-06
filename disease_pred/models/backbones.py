from typing import Dict

import torch
import torchvision.models as torchmodels
import transformers as tr
from torch import nn

from ..types.errors import ValueNotUnderstoodError
from .utils import ConvUnit


class PretrainedViT(nn.Module):
    """Pretrained Vision Transformer backbone."""

    def __init__(
        self,
        in_channels: int,
        out_classes: int,
        pretrained_name: str = "google/vit-base-patch16-224",
        **kwargs,
    ):
        super().__init__()
        self.vit = tr.ViTForImageClassification.from_pretrained(pretrained_name)
        self.vit.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.vit.vit.embeddings.patch_embeddings.projection.out_channels,
            kernel_size=self.vit.vit.embeddings.patch_embeddings.projection.kernel_size,
            stride=self.vit.vit.embeddings.patch_embeddings.projection.stride,
        )
        self.vit.classifier = nn.Linear(
            in_features=self.vit.classifier.in_features, out_features=out_classes, bias=self.vit.classifier.bias is not None
        )
        self.initial_conv = ConvUnit(
            in_channels=in_channels,
            out_channels=self.vit.vit.embeddings.patch_embeddings.projection.in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, pixel_values, *args, **kwargs):
        x = self.initial_conv(pixel_values)
        x = self.vit(pixel_values=x, *args, **kwargs)
        return x


class ViT(nn.Module):
    """Vision Transformer backbones ViT and DeiT.

    Uses huggingface implementation of ViT and DeiT
    (https://huggingface.co/docs/transformers/model_doc/)."""

    def __init__(self, params: Dict, pretrained: bool = False, **kwargs):
        vit_types = ["vit", "deit"]
        if params["vit_type"] not in vit_types:
            raise ValueNotUnderstoodError("ViT type", params["vit_type"], vit_types)
        output_layers = ["pooled", "last_hidden"]
        if params["output_layer"] not in output_layers:
            raise ValueNotUnderstoodError("Output layer argument", params["output_layer"], output_layers)

        super().__init__()
        self.output_layer = params["output_layer"]
        # Vision Transformer as joint backbone
        if not pretrained:
            if params["vit_type"] == "vit":
                config = tr.ViTConfig(**params)
                self.vit = tr.ViTModel(config, add_pooling_layer=params["output_layer"] == "pooled")
            elif params["vit_type"] == "deit":
                config = tr.DeiTConfig(**params)
                self.vit = tr.DeiTModel(config, add_pooling_layer=params["output_layer"] == "pooled")
        else:
            if params["vit_type"] == "vit":
                self.vit = PretrainedViT(
                    **params,
                    in_channels=params["num_channels"],
                    out_classes=params["hidden_size"],
                    output_layer=params["output_layer"],
                )
            else:
                raise NotImplementedError()

    def forward(self, x, *args, **kwargs):
        # Vision Transformer
        x = self.vit(pixel_values=x, *args, **kwargs)
        # ViT output = embedding of first (class) token
        if args or kwargs:
            return x
        else:
            if self.output_layer == "pooled":
                return x.pooler_output
            elif self.output_layer == "last_hidden":
                return x.last_hidden_state[:, 0, :]


class CVT(nn.Module):
    """Convolutional Vision Transformer (CVT) backbone.

    Uses huggingface implementation of CVT (https://huggingface.co/docs/transformers/model_doc/cvt).
    Paper: https://arxiv.org/abs/2103.15808"""

    def __init__(self, params: Dict, pretrained: bool = False, **kwargs):
        super().__init__()
        # Vision Transformer as joint backbone
        if not pretrained:
            self.config = tr.CvtConfig(**params)
            self.cvt = tr.CvtModel(self.config, add_pooling_layer=False)
            self.layernorm = nn.LayerNorm(self.config["embed_dim"][-1])
        else:
            raise NotImplementedError()

    def forward(self, x, *args, **kwargs):
        # Convolutional Vision Transformer
        x = self.cvt(pixel_values=x, *args, **kwargs)
        # CVT output = last hidden state and cls token value
        sequence_output = x.last_hidden_state
        if self.config["cls_token"][-1]:
            sequence_output = self.layernorm(x.cls_token_value)
        else:
            batch_size, num_channels, height, width = sequence_output.shape
            # rearrange "b c h w -> b (h w) c"
            sequence_output = sequence_output.view(batch_size, num_channels, height * width).permute(0, 2, 1)
            sequence_output = self.layernorm(sequence_output)

        sequence_output_mean = sequence_output.mean(dim=1)

        return sequence_output_mean


class CNN(nn.Module):
    """Convolutional Neural Network backbone."""

    def __init__(self, params: Dict, pretrained: bool = False, **kwargs):
        if pretrained:
            raise NotImplementedError()
        super().__init__()
        # optional CNN-based feature extractor
        self.cnn = nn.Sequential(
            ConvUnit(params["num_channels"], 16, 3, 1, 1),
            ConvUnit(16, 16, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvUnit(16, 32, 3, 1, 1),
            ConvUnit(32, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvUnit(32, 64, 3, 1, 1),
            ConvUnit(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvUnit(64, 128, 3, 1, 1),
            ConvUnit(128, 128, 3, 1, 1),
            ConvUnit(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvUnit(128, 128, 3, 1, 1),
            ConvUnit(128, 128, 3, 1, 1),
            ConvUnit(128, 128, 3, 1, 1),
        )
        self.HP = nn.Sequential(nn.MaxPool2d(2, 2), nn.AvgPool2d(kernel_size=3, stride=1))

    def forward(self, x, *args, **kwargs):
        # CNN
        x = self.cnn(x)
        # Hybrid pooling
        x = self.HP(x)
        x = x.view((x.size(0), -1))
        return x


class ResNet(nn.Module):
    """ResNet backbone with multiple depths."""

    def __init__(self, params: Dict, pretrained: bool = False, **kwargs):
        resnet_depths = [18, 34, 50, 101, 152]
        if params["depth"] not in resnet_depths:
            raise NotImplementedError(f"ResNet depth {params['depth']} unknown. Choose one of {resnet_depths}.")
        super().__init__()
        # ResNet model
        self.resnet = getattr(torchmodels, f"resnet{params['depth']}")(pretrained)
        self.resnet.conv1 = nn.Conv2d(
            in_channels=params["num_channels"], out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(in_features=self.resnet.fc.in_features, out_features=params["out_features"], bias=True)

    def forward(self, x, *args, **kwargs):
        x = self.resnet(x)
        return x


class VGG(nn.Module):
    """VGG backbone with different depths."""

    def __init__(self, params: Dict, pretrained: bool = False, **kwargs):
        if params["depth"] not in ["11", "11_bn", "13", "13_bn", "16", "16_bn", "19", "19_bn"]:
            raise NotImplementedError()
        super().__init__()
        # ResNet model
        self.vgg = getattr(torchmodels, f"vgg{params['depth']}")(pretrained)
        self.vgg.features[0] = nn.Conv2d(
            in_channels=params["num_channels"], out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.vgg.classifier = nn.Linear(
            in_features=self.vgg.classifier[0].in_features, out_features=params["out_features"], bias=True
        )

    def forward(self, x, *args, **kwargs):
        x = self.vgg(x)
        return x


class CVAE(nn.Module):
    """Convolutional Variational Autoencoder backbone."""

    def __init__(self, params: Dict, pretrained: bool = False, **kwargs):
        if pretrained:
            raise NotImplementedError()
        super().__init__()
        if params["image_size"] % 4 != 0:
            print("WARNING: Image size is not divisible by 4. " "The decoded image will not have the same size.")

        # init encoder layers
        hybrid_pooling = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.AvgPool2d(kernel_size=3, stride=1, padding=1))
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(params["num_channels"], 16, 3, 1, 1),
            nn.ReLU(True),
            hybrid_pooling,
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(True),
            hybrid_pooling,
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            hybrid_pooling,
        )
        with torch.no_grad():
            conv_shape = self.encoder_conv(
                torch.empty(1, params["num_channels"], params["image_size"], params["image_size"])
            ).shape[1:]
        flattened_dim = torch.prod(torch.as_tensor(conv_shape))
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin_mean = nn.Linear(flattened_dim, params["latent_dim"])
        self.encoder_lin_std = nn.Linear(flattened_dim, params["latent_dim"])

        # init decoder layers
        self.decoder_lin = nn.Linear(params["latent_dim"], flattened_dim)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=conv_shape)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2, 0, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 2, 2, 0, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, params["num_channels"], 2, 2, 0, 0, bias=False),
        )

        # binary cross entropy loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def encoder(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mean = self.encoder_lin_mean(x)
        log_var = self.encoder_lin_std(x)
        return {"mean": mean, "log_var": log_var}

    def decoder(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

    def reparameterize(self, input):
        std = torch.exp(input["log_var"] / 2)
        eps = torch.randn_like(std)
        return input["mean"] + std * eps

    def forward(self, x, *args, **kwargs):
        enc_out = self.encoder(x)
        z = self.reparameterize(enc_out)
        out = self.decoder(z)
        return {**enc_out, "out": out}

    def calculate_loss(self, x, output):
        # KL divergence
        kl_div = torch.mean(
            -0.5 * torch.sum(1 + output["log_var"] - output["mean"] ** 2 - output["log_var"].exp(), dim=1), dim=0
        )
        # BCE loss (sigmoid already included)
        bin_ce = self.bce_loss(output["out"], x)
        return {"kl_div_loss": kl_div.mean(), "bin_ce_loss": bin_ce}


if __name__ == "__main__":
    pass
