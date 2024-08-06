from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import nn, optim

from .. import MODEL_DIR
from ..types.errors import ValueNotUnderstoodError
from ..utils import instantiate_module
from . import backbones, preprocessing


class MLPNeck(nn.Module):
    """Implementation of a Multi-layer Perceptron Neck for the MultiDLDL model"""

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        act_func: str = "relu",
        norm: str = "layer",
        dropout_prob: float = 0.02,
        pretrained: bool = False,
        freeze_layers: bool = False,
        **kwargs,
    ):
        if act_func not in ["linear", "relu", "gelu"]:
            raise NotImplementedError()
        if norm not in ["layer", "batch", "none"]:
            raise NotImplementedError()
        if pretrained:
            raise NotImplementedError()
        if freeze_layers:
            raise NotImplementedError()
        super().__init__()
        ls = [in_features] + layer_sizes
        mlp_layers = []
        for step, (this_ls, next_ls) in enumerate(zip(ls[:-1], ls[1:])):
            mlp_layers.append(nn.Linear(in_features=this_ls, out_features=next_ls, bias=True))
            if step < len(layer_sizes) - 1:
                if norm == "layer":
                    mlp_layers.append(nn.LayerNorm(next_ls))
                elif norm == "batch":
                    mlp_layers.append(nn.BatchNorm1d(next_ls))
                if act_func == "relu":
                    mlp_layers.append(nn.ReLU())
                elif act_func == "gelu":
                    mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(p=dropout_prob))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        return self.mlp(x)


class MultiLDLHead(nn.Module):
    """Label distribution learning module with multiple heads"""

    def __init__(
        self,
        in_features: int,
        quantization_steps: List[int],
        individual_mlp: Optional[Dict[str, Any]] = None,
        norm: str = "layer",
        pretrained: bool = False,
        freeze_layers: bool = False,
        **kwargs,
    ):
        if norm not in ["layer", "batch", "none"]:
            raise NotImplementedError()
        if pretrained:
            raise NotImplementedError()
        if freeze_layers:
            raise NotImplementedError()
        super().__init__()

        self.n_heads = len(quantization_steps)

        # if config for label-specific mlps is given, initialize them and mix them in the last layer
        if individual_mlp is not None:
            necks = []
            individual_mlp["in_features"] = in_features
            for _ in quantization_steps:
                necks.append(MLPNeck(**individual_mlp))

            self.idv_mlps = nn.ModuleList(necks)
            in_features = individual_mlp["layer_sizes"][-1]

            # label mixing
            # convolution of the embeddings of each label for the distinct label heads
            self.label_mix = nn.Conv1d(in_channels=self.n_heads, out_channels=self.n_heads, kernel_size=1, bias=False)
            # self.label_mix.weight.data = (torch.ones((self.n_heads, self.n_heads)) * 0.5 + torch.eye(self.n_heads) * 0.5)[
            #    :, :, None
            # ]
            self.label_mix.weight.data = torch.eye(self.n_heads)[:, :, None]
        else:
            self.idv_mlps = None
            self.label_mix = None

        heads = []
        for steps in quantization_steps:
            modules = []
            modules.append(nn.Linear(in_features=in_features, out_features=steps, bias=False))
            if norm == "layer":
                modules.append(nn.LayerNorm(steps))
            elif norm == "batch":
                modules.append(nn.BatchNorm1d(steps))
            modules.append(nn.Softmax(-1))
            heads.append(nn.Sequential(*modules))

        self.ldl = nn.ModuleList(heads)

    def forward(self, x, idx):
        if self.idv_mlps is not None:
            # collect outputs from all necks
            all_neck_outputs = []
            for i in range(self.n_heads):
                if i == idx:
                    all_neck_outputs.append(self.idv_mlps[idx](x))
                else:
                    with torch.no_grad():
                        all_neck_outputs.append(self.idv_mlps[i](x))
            # stack outputs
            all_neck_outputs = torch.stack(all_neck_outputs, dim=1)

            # convolve with label relevance layer
            output = self.label_mix(all_neck_outputs)[:, idx, :]

            # return ldl head output
            return self.ldl[idx](output)
        else:
            # return ldl head output
            return self.ldl[idx](x)


# Gao, B. Bin, Xing, C., Xie, C. W., Wu, J., & Geng, X. (2017).
# Deep Label Distribution Learning with Label Ambiguity.
# IEEE Transactions on Image Processing, 26(6), 2825–2838.
# https://doi.org/10.1109/TIP.2017.2689998
class MultiDLDL(pl.LightningModule):
    """Model for Multihead Deep Label Distribution Learning with custom backbones."""

    def __init__(self, config: Dict[str, Any]):
        """Initializes MultiDLDL module.

        Args:
            config (Dict[str, Any]): Model configuration

        Raises:
            ValueError: If number of given LDL parameters are uncompatible.

        """
        super().__init__()

        # validate config
        if not len(config["ldl"]["reg_limits"]) == len(config["ldl"]["quantization_steps"]) == len(config["ldl"]["sigmas"]):
            raise ValueError(
                "Number of regression limits, quantizations steps, and label distribution std do not fit to each other."
            )

        # save hyperparameters
        self.save_hyperparameters(config)

        # deactivate automatic optimization
        self.automatic_optimization = False

        # set attributes used in the methods
        self.labels = self.hparams.ldl["labels"]
        self.n_steps = self.hparams.ldl["quantization_steps"]
        self.reg_limits = self.hparams.ldl["reg_limits"]
        self.sigmas = self.hparams.ldl["sigmas"]
        self.norm_sigmas = [
            self.sigmas[i] / (self.reg_limits[i][1] - self.reg_limits[i][0]) if self.sigmas[i] else None
            for i in range(len(self.sigmas))
        ]
        self.steps = [torch.linspace(*rl, steps=n) for (rl, n) in zip(self.reg_limits, self.n_steps)]
        self.norm_steps = [torch.linspace(0, 1, steps=n) for n in self.n_steps]
        self.loss_method = self.hparams.optimizer["loss_method"]
        # optional attributes
        if "params" in self.hparams.backbone.get("args", []):
            self.image_size = self.hparams.backbone["args"]["params"].get("image_size")
            self.image_resolution = self.hparams.backbone["args"]["params"].get("image_resolution")
            self.image_channels = self.hparams.backbone["args"]["params"].get("image_channels")

        # configure data preprocessor:
        # remark: this internal preprocessor is only used in inference mode
        # the training data has to be preprocessed via the (external) dataloader
        if self.hparams.get("normalizer") is not None:
            self.preprocessor = instantiate_module(preprocessing, self.hparams.normalizer)
        else:
            self.preprocessor = None

        # configure backbone
        # load checkpoint if available
        if self.hparams.backbone.get("checkpoint") is not None:
            self.backbone, backbone_config = load_multidldl_submodel(
                Path(self.hparams.backbone["checkpoint"]), "backbone", "cpu"
            )
            # freeze layers if desired
            if self.hparams.backbone["freeze_layers"]:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            self.hparams.backbone.update(backbone_config)
        # else, initialize new
        else:
            self.backbone = instantiate_module(backbones, self.hparams.backbone)
            # apply weight initialization if not backbone is not pretrained
            self.backbone.apply(self._init_weights)

        # optional MLP between backbone and LDL head
        if self.hparams.get("mlp") is not None:
            # load checkpoint if available
            if self.hparams.mlp.get("checkpoint") is not None:
                self.mlp, mlp_config = load_multidldl_submodel(Path(self.hparams.mlp["checkpoint"]), "mlp", "cpu")
                # freeze layers if desired
                if self.hparams.mlp["freeze_layers"]:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                self.hparams.mlp.update(mlp_config)
            # else, initialize new
            else:
                self.mlp = MLPNeck(**self.hparams.mlp)
                self.mlp.apply(self._init_weights)
        else:
            self.mlp = None

        # deep label distribution learning (DLDL) heads
        self.ldl = MultiLDLHead(**self.hparams.ldl)
        self.ldl.apply(self._init_weights)

        # define losses
        # -> KL divergence loss for distribution learning
        self.kl_loss = nn.KLDivLoss(reduction="none", log_target=False)

        # -> L1 loss for expectation value (used in conventional method)
        if self.hparams.optimizer["loss_method"] == "conventional":
            self.l1_loss = nn.L1Loss(reduction="none")

    @staticmethod
    def _init_weights(layer: nn.Module):
        """Initializes weights with Xavier uniform distribution or zeros for biases.

        Args:
            layer: Module layer.
        """
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if isinstance(layer, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, optimizer_idx: int, **kwargs) -> Dict[str, torch.Tensor]:
        dists = kwargs.get("dists", None)

        if not x.device == self.device:
            raise RuntimeError(
                f"Input on {x.device} and model on {self.device}. Please assure, that input and model are on the same device."
            )
        # preprocess batch
        if (not self.training) and self.preprocessor:
            valid_mask = kwargs.get("valid_mask", None)
            x = self.preprocessor(batch=x, valid_mask=valid_mask)

        # in case of CVAE backbone
        if isinstance(self.backbone, backbones.CVAE):
            mean, log_var, out = self.backbone(x)
            # optimizer for encoder/decoder
            if optimizer_idx == 0:
                return {"mean": mean, "log_var": log_var, "out": out}
            # optimizer for DLDL
            else:
                if self.mlp:
                    x = self.mlp(mean)
                else:
                    x = mean
                if optimizer_idx > 0:
                    # backbone/MLP output -> Multi-LDL-Head
                    ldl_output = self.ldl(x=x, idx=optimizer_idx - 1)  # pdf
                    # expectation value for corresponding LDL output (normalized to [0,1])
                    exp_output_norm = torch.sum(
                        ldl_output * self.norm_steps[optimizer_idx - 1].to(ldl_output.device, non_blocking=True), dim=1
                    )
                elif optimizer_idx == -1:
                    # joint forward pass of all LDL heads
                    with torch.no_grad():
                        ldl_output = [self.ldl(x=x, idx=idx) for idx in range(len(self.sigmas))]
                        exp_output_norm = [
                            torch.sum(ldl_out * self.norm_steps[idx].to(ldl_out.device, non_blocking=True), dim=1)
                            for idx, ldl_out in enumerate(ldl_output)
                        ]
                else:
                    raise ValueError(f"optimizer_idx has wrong value ({optimizer_idx}).")

                return {"label_dist": ldl_output, "exp_norm": exp_output_norm}

        # in case of other backbone
        else:
            x = self.backbone(x)
            if self.mlp:
                if isinstance(dists, torch.Tensor):
                    cat_x = torch.cat((x, dists), dim=1)
                    x = self.mlp(cat_x)
                else:
                    x = self.mlp(x)
            if optimizer_idx >= 0:
                # backbone/MLP output -> Multi-LDL-Head
                ldl_output = self.ldl(x=x, idx=optimizer_idx)  # pdf
                # expectation value for corresponding LDL output (normalized to [0,1])
                exp_output_norm = torch.sum(
                    ldl_output * self.norm_steps[optimizer_idx].to(ldl_output.device, non_blocking=True), dim=1
                )
            elif optimizer_idx == -1:
                # joint forward pass of all LDL heads
                with torch.no_grad():
                    ldl_output = [self.ldl(x=x, idx=idx) for idx in range(len(self.sigmas))]
                    exp_output_norm = [
                        torch.sum(ldl_out * self.norm_steps[idx].to(ldl_out.device, non_blocking=True), dim=1)
                        for idx, ldl_out in enumerate(ldl_output)
                    ]
            else:
                raise ValueError(f"optimizer_idx has wrong value ({optimizer_idx}).")
            return {"label_dist": ldl_output, "exp_norm": exp_output_norm}

    def get_label_relevance_matrix(self) -> Optional[torch.Tensor]:
        """Get the label relevace matrix given by the weights of the convolution layer
        after the individual MLPs for the labels in the LDL head.

        Returns:
            Optional[torch.Tensor]: Label relevance matrix.
        """
        if self.ldl.label_mix is not None:
            return list(self.ldl.label_mix.parameters())[0].data[:, :, 0]
        else:
            return None

    def configure_optimizers(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Configures the optimizers and corresponding (optional) learning rate schedulers.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Optimizer and learning rate scheduler list.
        """
        opt_list = []
        sch_list = []
        if isinstance(self.backbone, backbones.CVAE):
            optimizer = getattr(optim, self.hparams.optimizer["module"])(self.parameters(), **self.hparams.optimizer["args"])
            opt_list.append(optimizer)
            if "lr_scheduler" in self.hparams:
                lr_scheduler = {
                    "scheduler": getattr(optim.lr_scheduler, self.hparams.lr_scheduler["module"])(
                        optimizer, **self.hparams.lr_scheduler["args"]
                    ),
                    "interval": self.hparams.lr_scheduler["interval"],
                    "monitor": "train_loss_cvae",
                    "name": "lr_cvae",
                }
                sch_list.append(lr_scheduler)

        for i in range(len(self.reg_limits)):
            optimizer = getattr(optim, self.hparams.optimizer["module"])(self.parameters(), **self.hparams.optimizer["args"])
            opt_list.append(optimizer)
            if "lr_scheduler" in self.hparams:
                lr_scheduler = {
                    "scheduler": getattr(optim.lr_scheduler, self.hparams.lr_scheduler["module"])(
                        optimizer, **self.hparams.lr_scheduler["args"]
                    ),
                    "interval": self.hparams.lr_scheduler["interval"],
                    "monitor": f"train_loss_{self.labels[i]}",
                    "name": f"lr_{self.labels[i]}",
                }
                sch_list.append(lr_scheduler)

        return opt_list, sch_list

    def calculate_loss(
        self, output: Dict[str, torch.Tensor], target: Dict[str, Optional[torch.Tensor]], idx: int, reduction: str = "mean"
    ) -> Dict[str, torch.Tensor]:
        """Calculates loss with convertional or Full-KL-Loss method (https://arxiv.org/abs/2209.02055).

        Args:
            output (Dict[str, torch.Tensor]): Dictionary with model output.
            target (Dict[str, Optional[torch.Tensor]]): Dictionary with ground truth targets.
            idx (int): Multi-Head index.
            reduction (str, optional): Reduction method. Defaults to "mean".

        Raises:
            ValueError: If loss or reduction method is unknown.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with loss values.
        """
        calc_methods = ["full_KL", "conventional"]
        if self.loss_method not in calc_methods:
            raise ValueNotUnderstoodError("Loss method", self.loss_method, calc_methods)

        reduction_methods = ["mean", "sum", "none"]
        if reduction not in reduction_methods:
            raise ValueNotUnderstoodError("Reduction method", reduction, reduction_methods)

        # new full-KL loss without hyperparameter
        # Loss = ldl_loss + exp_loss + smooth_penalty
        if self.loss_method == "full_KL":
            # LDL loss: KL divergence between output distributions
            ldl = self.kl_loss(torch.log(output["label_dist"]), target["label_dist"]).sum(dim=1)

            # Expectation value loss: KL divergence between
            # true dist and pred dist, if pred dist would be gaussian
            pred_var_norm = torch.sum(
                output["label_dist"]
                * torch.pow(
                    self.norm_steps[idx].to(output["label_dist"].device, non_blocking=True) - output["exp_norm"][:, None], 2
                ),
                dim=1,
            )
            if target["stds"] is None:
                target_var_norm = self.norm_sigmas[idx] ** 2
            else:
                target_var_norm = torch.pow(
                    (target["stds"] / (self.reg_limits[idx][1] - self.reg_limits[idx][0])).to(
                        output["label_dist"].device, non_blocking=True
                    ),
                    2,
                )
            exp = (
                0.5 * torch.log(pred_var_norm / target_var_norm)
                + (target_var_norm + torch.pow(output["exp_norm"] - target["exp_norm"], 2)) / (2.0 * pred_var_norm)
                - 0.5
            )

            # Smoothness loss: (symmetric) KL divergence between pred dist and its shifted version (by 1 point)
            p, q = output["label_dist"][:, 1:], output["label_dist"][:, :-1]
            p = p / p.sum(dim=1)[:, None]
            q = q / q.sum(dim=1)[:, None]
            kl_pq_qp = (p - q) * torch.log(p / q)  # = KL(p,q) + KL (q,p) (ohne Summe)
            kl_pq_qp[torch.isclose(q, p)] = 0  # avoids numerical issues
            smooth_penalty = kl_pq_qp.sum(dim=1) * 0.5

            # apply weights if given in the data
            if isinstance(target["weights"], torch.Tensor):
                weights_sum = target["weights"].sum()
                ldl = (ldl * target["weights"][:, None]) / weights_sum
                exp = (exp * target["weights"][:, None]) / weights_sum
                smooth_penalty = (smooth_penalty * target["weights"][:, None]) / weights_sum

            if reduction == "mean":
                return {"ldl": ldl.mean(), "exp": exp.mean(), "smooth": smooth_penalty.mean()}
            elif reduction == "sum":
                return {"ldl": ldl.sum(), "exp": exp.sum(), "smooth": smooth_penalty.sum()}
            elif reduction == "none":
                return {"ldl": ldl, "exp": exp, "smooth": smooth_penalty}

        # conventional loss with hyperparameter
        # Loss = ldl_loss (KL) + exp_loss (L1)
        elif self.loss_method == "conventional":
            # LDL loss: KL divergence between output distributions
            ldl = self.kl_loss(torch.log(output["label_dist"]), target["label_dist"]).sum(dim=1)

            # Expectation value loss: L1 loss
            exp = self.l1_loss(self.scale_from_norm(output["exp_norm"], idx), self.scale_from_norm(target["exp_norm"], idx))

            # apply weights if given in the data
            if isinstance(target["weights"], torch.Tensor):
                weights_sum = target["weights"].sum()
                ldl = (ldl * target["weights"][:, None]) / weights_sum
                exp = (exp * target["weights"][:, None]) / weights_sum
                smooth_penalty = (smooth_penalty * target["weights"][:, None]) / weights_sum

            if reduction == "mean":
                return {"ldl": ldl.mean(), "exp": exp.mean()}
            elif reduction == "sum":
                return {"ldl": ldl.sum(), "exp": exp.sum()}
            elif reduction == "none":
                return {"ldl": ldl, "exp": exp}

    def calculate_metrics(
        self, output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor], idx: int
    ) -> Dict[str, torch.Tensor]:
        """Calculates further model evaluation metrics.

        Args:
            output (Dict[str, torch.Tensor]): Dictionary with model output.
            target (Dict[str, Optional[torch.Tensor]]): Dictionary with ground truth targets.
            idx (int): Multi-Head index.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with metrics.
        """
        metrics = {}
        pred_exp = self.scale_from_norm(output["exp_norm"], idx)
        target_exp = self.scale_from_norm(target["exp_norm"], idx)

        metrics["MAE"] = torch.abs(pred_exp - target_exp).mean()
        # Mean Distribution Overlap (MDO) = shared integral of ground truth and prediction label distribution
        metrics["MDO"] = torch.min(output["label_dist"], target["label_dist"]).sum() / len(target["label_dist"])
        return metrics

    def calculate_confidence(self, output: torch.Tensor, idx: int) -> Dict[str, torch.Tensor]:
        """Calculates the prediction confidence of the model.

        Args:
            output (torch.Tensor): Dictionary with model output.
            idx (int): Multi-Head index.

        Returns:
            torch.Tensor: Prediction confidence of the model.
        """
        sigma_pred = (
            torch.sum(
                output["label_dist"]
                * (self.norm_steps[idx].to(output["label_dist"].device, non_blocking=True) - output["exp_norm"][:, None]) ** 2,
                axis=1,
            )
            ** 0.5
        )
        sigma_true = self.norm_sigmas[idx]

        conf = sigma_true / sigma_pred
        # interpretation of the confidence value:
        # conf in (0, inf)
        # conf = 1: optimal confidence = predicted std is equal to trained std
        # conf > 1: overconfidence = predicted std is lower than trained std by corresponding factor
        # conf < 1: underconfidence = trained std is lower than predicted std by corresponding factor

        return {"confidence": conf, "sigma_pred": sigma_pred * (self.reg_limits[idx][1] - self.reg_limits[idx][0])}

    def make_gaussian_label_dist(self, mu: torch.Tensor, idx: int, sig: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generates (discrete) Gaussian distributions from fiven means and stds.

        Args:
            mu (torch.Tensor): Mean values.
            idx (int): Multi-Head index.
            sig (Optional[torch.Tensor], optional): Std values. Defaults to None.
                If None, the stds are given by the model configuration of the respective LDL head.

        Returns:
            torch.Tensor: Tensor with discrete Gaussion distributions
        """
        # normal distribution around target value
        if sig is None:
            sig = self.sigmas[idx]
        else:
            sig = sig.to(mu.device, non_blocking=True)[:, None]
        p_k = torch.exp(-((self.steps[idx].to(mu.device, non_blocking=True) - mu[:, None]) ** 2) / (2.0 * sig**2))
        p_k = p_k / p_k.sum(dim=1)[:, None]
        return p_k

    def valid_labels(self, labels: torch.Tensor, idx: int) -> torch.Tensor:
        """Returns a mask of valid labels.

        Args:
            labels (torch.Tensor): Tensor with labels.
            idx (int): Multi-Head index.

        Returns:
            torch.Tensor: Mask of valid labels.
        """
        flabels = labels.type(torch.float)
        high = flabels <= self.reg_limits[idx][1]
        low = self.reg_limits[idx][0] <= flabels
        return high & low

    def scale_to_norm(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Scales original label range to normalized range [0, 1].

        Args:
            x (torch.Tensor): Tensor to normalize.
            idx (int): Multi-Head index.

        Returns:
            torch.Tensor: Scaled (normalized) tensor.
        """
        return (x - self.reg_limits[idx][0]) / (self.reg_limits[idx][1] - self.reg_limits[idx][0])

    def scale_from_norm(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """Recales normalized label range to original label range.

        Args:
            x (torch.Tensor): Tensor to rescale.
            idx (int): Multi-Head index.

        Returns:
            torch.Tensor: Rescaled tensor.
        """
        return x * (self.reg_limits[idx][1] - self.reg_limits[idx][0]) + self.reg_limits[idx][0]

    def prepare_target(
        self, data_targets: Dict[str, Any], idx: int, device: torch.device = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Prepares targets (labels, valid masks, weights, stds, ...) for processing in model.

        Args:
            data_targets (Dict[str, Any]): Targets output of DataLoader.
            idx (int): Multi-Head index.
            device (torch.device, optional): Device for calculation. Defaults to None.

        Returns:
            Dict[str, Optional[torch.Tensor]]: Prepared targets dictionary.
        """
        if not device:
            device = self.device
        valid = self.valid_labels(data_targets["labels"][idx], idx)

        # single value label is given -> make Gaussian label distribution with given label as mu and predefined std
        if data_targets["labels"][idx].dim() == 1:
            target_exp = data_targets["labels"][idx][valid]
            target = {
                "label_dist": self.make_gaussian_label_dist(
                    mu=target_exp,
                    sig=data_targets["stds"][idx][valid] if "stds" in data_targets.keys() else None,
                    idx=idx,
                ).to(device, non_blocking=True),
                "exp_norm": self.scale_to_norm(target_exp, idx).to(device, non_blocking=True),
                "valid": valid.to(device, non_blocking=True),
                # "vi_hist": data_targets["vi_hist"],
                "weights": (
                    (data_targets["weight"][idx][valid]).to(device, non_blocking=True)
                    if "weight" in data_targets.keys()
                    else None
                ),
                "stds": (
                    (data_targets["stds"][idx][valid]).to(device, non_blocking=True) if "stds" in data_targets.keys() else None
                ),
            }

        # distribution is given directly
        elif data_targets["labels"][idx].dim() == 2:
            label_dist = data_targets["labels"][idx]
            # normalize label dists
            label_dist = label_dist / label_dist.sum(dim=1)[:, None]
            target_exp = torch.sum(label_dist * self.steps[idx].to(device, non_blocking=True), dim=1)
            target_std = torch.pow(
                torch.sum((label_dist * torch.pow(self.steps[idx].to(device, non_blocking=True) - target_exp[:, None], 2))), 0.5
            )
            target = {
                "label_dist": label_dist,
                "exp_norm": self.scale_to_norm(target_exp, idx),
                "valid": torch.ones(len(label_dist), dtype=bool),
                # "vi_hist": data_targets["vi_hist"],
                "weights": None,
                "stds": target_std,
            }
        else:
            raise ValueError("Label data type not understood.")

        return target

    def _process_input(
        self, images: torch.Tensor, targets: Dict[str, torch.Tensor], optimizer_idx: int
    ) -> Dict[str, torch.Tensor]:
        target = self.prepare_target(targets, optimizer_idx)
        output = self(images[target["valid"]], optimizer_idx=optimizer_idx)  # , dists=target["vi_hist"][target["valid"]])

        # DLDL loss
        loss_dict = self.calculate_loss(output, target, optimizer_idx)

        # further metrics
        metrics_dict = self.calculate_metrics(output, target, optimizer_idx)

        return {"losses": loss_dict, "metrics": metrics_dict}

    @staticmethod
    def _check_nan(
        loss,
        images: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        optimizer_idx: Optional[int] = None,
        raise_error: bool = True,
    ) -> None:
        if not torch.isfinite(loss):
            error_string = ""
            if optimizer_idx is not None:
                error_string += f"Optimizer No. {optimizer_idx} -> Loss is NaN -> targets OK?: {torch.all(torch.isfinite(targets['labels'][optimizer_idx]))}, images OK?: {torch.all(torch.isfinite(images))} (min {images.min().item()}, max {images.max().item()})"
            else:
                error_string += f"Loss is NaN -> targets OK?: {torch.all(torch.isfinite(torch.as_tensor(targets['labels'])))}, images OK?: {torch.all(torch.isfinite(images))} (min {images.min().item()}, max {images.max().item()})"
            if (not torch.all(torch.isfinite(images))) and ("dataset" in targets.keys()):
                error_string += f"\naffected datasets: {set([d for d, is_f in zip(targets['dataset'], ~torch.isfinite(images).all(dim=1).all(dim=1).all(dim=1)) if is_f])}"
            if raise_error:
                raise ValueError(error_string)
            else:
                print(error_string)
        else:
            pass

    def training_step(self, batch: Tuple, batch_idx: int):
        images, targets = batch

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if schedulers is None:
            schedulers = [None] * len(optimizers)
        elif not isinstance(schedulers, list):
            schedulers = [schedulers]

        res = {}

        for idx, (opt, sch) in enumerate(zip(optimizers, schedulers)):
            if isinstance(self.backbone, backbones.CVAE):
                # train encoder/decoder
                if idx == 0:
                    output = self(images, idx)

                    # CVAE loss
                    loss = self.backbone.calculate_loss(images, output)
                    cvae_loss = loss["kl_div_loss"] + loss["bin_ce_loss"]

                    opt.zero_grad()
                    self.manual_backward(cvae_loss)
                    opt.step()

                    self.log("train_loss_cvae", cvae_loss, on_epoch=True, sync_dist=True)

                    self.log("global_step", float(self.global_step))

                    return {"loss": cvae_loss}

                else:
                    idx -= 1

            # train DLDL
            output = self._process_input(images, targets, idx)
            loss_sum = sum(v for (k, v) in output["losses"].items())

            self._check_nan(loss_sum, images, targets, idx)

            # manual optimization
            opt.zero_grad()
            self.manual_backward(loss_sum)
            opt.step()

            # step lr schedulers
            if sch:
                if self.hparams.lr_scheduler["interval"] == "step":
                    sch.step()
                elif self.hparams.lr_scheduler["interval"] == "epoch":
                    if self.trainer.is_last_batch:
                        sch.step()

            self.log(f"train_loss_{self.labels[idx]}", loss_sum, on_epoch=True, sync_dist=True)
            for k, v in output["losses"].items():
                self.log(f"train_loss_{k}_{self.labels[idx]}", v, on_epoch=True, sync_dist=True)
            for k, v in output["metrics"].items():
                self.log(f"train_{k}_{self.labels[idx]}", v, on_epoch=True, sync_dist=True)
            rel_mat = self.get_label_relevance_matrix()
            if rel_mat is not None:
                for i, cross_label in enumerate(self.labels):
                    self.log(f"train_rel_{cross_label}->{self.labels[idx]}", rel_mat[idx, i])
            self.log("global_step", float(self.global_step), rank_zero_only=True)

            res[f"train_loss_{self.labels[idx]}"] = loss_sum

        return res

    def on_train_epoch_end(self):
        self.log("epoch", float(self.current_epoch), rank_zero_only=True, sync_dist=True)

    @torch.no_grad()
    def validation_step(self, batch: Tuple, batch_idx: int, dataloader_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # ordinary validation
        if dataloader_idx is None or dataloader_idx == 0:
            images, targets = batch

            losses = []
            sum_losses = []

            if isinstance(self.backbone, backbones.CVAE):
                # validate encoder/decoder
                output = self(images, 0)

                # CVAE loss
                loss = self.backbone.calculate_loss(images, output)
                cvae_loss = loss["kl_div_loss"] + loss["bin_ce_loss"]

                losses.append([loss["kl_div_loss"], loss["bin_ce_loss"]])
                sum_losses.append(cvae_loss)

                self.log("val_loss_cvae", cvae_loss, add_dataloader_idx=False, sync_dist=True)

            # validate DLDLs
            for idx in range(len(self.reg_limits)):
                output = self._process_input(images, targets, idx)
                loss_sum = sum(v for (k, v) in output["losses"].items())

                losses.append(list(output["losses"].values()))
                sum_losses.append(loss_sum)

                self._check_nan(loss_sum, images, targets, idx, raise_error=False)

                self.log(f"val_loss_{self.labels[idx]}", loss_sum, sync_dist=True, add_dataloader_idx=False)
                for k, v in output["losses"].items():
                    self.log(f"val_loss_{k}_{self.labels[idx]}", v, sync_dist=True, add_dataloader_idx=False)
                for k, v in output["metrics"].items():
                    self.log(f"val_{k}_{self.labels[idx]}", v, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

            loss = torch.sum(torch.stack(sum_losses, dim=0), dim=0)
            self.log("val_loss", loss, sync_dist=True, add_dataloader_idx=False)

            return {"val_loss": loss}

        # online test
        elif dataloader_idx == 1:  # and self.current_epoch % self.hparams.train.test_every_n_epoch == 0:
            self.test_step(batch, batch_idx)

    @torch.no_grad()
    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        images, targets = batch

        losses = []
        sum_losses = []

        if isinstance(self.backbone, backbones.CVAE):
            # test encoder/decoder
            output = self(images, 0)

            # CVAE loss
            loss = self.backbone.calculate_loss(images, output)
            cvae_loss = loss["kl_div_loss"] + loss["bin_ce_loss"]

            losses.append([loss["kl_div_loss"], loss["bin_ce_loss"]])
            sum_losses.append(cvae_loss)

            self.log("test_loss_cvae", cvae_loss, add_dataloader_idx=False)

        # test DLDLs
        for idx in range(len(self.reg_limits)):
            output = self._process_input(images, targets, idx)
            loss_sum = sum(v for (k, v) in output["losses"].items())

            losses.append(list(output["losses"].values()))
            sum_losses.append(loss_sum)

            self._check_nan(loss_sum, images, targets, idx, raise_error=False)

            self.log(f"test_loss_{self.labels[idx]}", loss_sum, sync_dist=True, add_dataloader_idx=False)
            for k, v in output["losses"].items():
                self.log(f"test_loss_{k}_{self.labels[idx]}", v, sync_dist=True, add_dataloader_idx=False)
            for k, v in output["metrics"].items():
                self.log(f"test_{k}_{self.labels[idx]}", v, prog_bar=True, sync_dist=True, add_dataloader_idx=False)

        loss = torch.sum(torch.stack(sum_losses, dim=0), dim=0)
        self.log("test_loss", loss, sync_dist=True, add_dataloader_idx=False)

        return {"test_loss": loss}


def load_multidldl_submodel(path: Path, submodel_name: str, map_location: str = None) -> Tuple[nn.Module, Dict]:
    """Loads submodel from MultiDLDL model from a checkpoint in the given path.

    Args:
        path (str): Path to checkpoint. If path can be found directly, it is used.
            Otherwise, the path is interpreted relative to the package's model directory.
        submodel_name (str): Name of the submodel.

    Raises:
        ValueError: If path cannot be found.

    Returns:
        nn.Module: MultiDLDL submodel from checkpoint and corresponding config.
    """
    path = path if path.exists() else MODEL_DIR / path
    if path.exists():
        main_module = MultiDLDL.load_from_checkpoint(path, map_location)
        submodule_config = main_module.hparams.get(submodel_name)
        return getattr(main_module, submodel_name), submodule_config
    else:
        raise ValueError(f"Path to submodel {submodel_name} not found: {path}")
