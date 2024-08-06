from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from skimage.exposure import equalize_hist
from torch import nn


class Preprocessor(nn.Module):
    """Base class for preprocessor modules."""

    def __init__(self) -> None:
        super().__init__()

    def combine_valid_masks_torch(self, batch: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = torch.all(torch.isfinite(batch), dim=-3)
        elif not valid_mask.shape == batch[:, -1, :, :].shape:
            raise ValueError(f"valid_mask has wrong shape {valid_mask.shape}. Should be {batch[:,-1,:,:].shape}.")

        valid_mask = valid_mask[:, None, :, :] & torch.isfinite(batch)

        return valid_mask

    def combine_valid_masks_numpy(self, batch: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if valid_mask is None:
            valid_mask = np.all(np.isfinite(batch), axis=-3)
        elif not valid_mask.shape == batch[:, -1, :, :].shape:
            raise ValueError(f"valid_mask has wrong shape {valid_mask.shape}. Should be {batch[:,-1,:,:].shape}.")

        valid_mask = valid_mask[:, None, :, :] & np.isfinite(batch)

        return valid_mask

    def mask_batch_torch(
        self, batch: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, invalid_value: Optional[Any] = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = self.combine_valid_masks_torch(batch=batch, valid_mask=valid_mask)
        masked_batch = torch.where(valid_mask, batch, torch.empty(1, device=batch.device).fill_(invalid_value))

        return masked_batch, valid_mask

    def mask_batch_numpy(
        self, batch: torch.Tensor, valid_mask: Optional[torch.Tensor] = None, invalid_value: Optional[Any] = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_mask = self.combine_valid_masks_numpy(batch=batch, valid_mask=valid_mask)
        masked_batch = np.where(valid_mask, batch, np.empty(1) * invalid_value)

        return masked_batch, valid_mask


class Normalizer(Preprocessor):
    """Preprocessing module that performs image normalization."""

    def __init__(
        self,
        mean: List[float],
        std: List[float],
        clip: Optional[Tuple[Union[float, int], Union[float, int]]] = None,
        **kwargs,
    ):
        """
        Args:
            mean (torch.Tensor): Predifined mean for each channel [C].
            std (torch.Tensor): Predifined std for each channel [C].
            clip (Optional[Tuple[Union[float, int], Union[float, int]]], optional): If given, clip output to this range.
                Defaults to None.
        """
        super().__init__()
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.clip = clip

    def forward(self, batch: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        masked_batch, valid_mask = self.mask_batch_torch(batch, valid_mask, 0)

        norm_batch = torch.divide(
            masked_batch - self.mean[None, :, None, None].to(batch.device, non_blocking=True),
            self.std[None, :, None, None].to(batch.device, non_blocking=True),
        )

        norm_batch = torch.where(valid_mask, norm_batch, 0)

        return norm_batch.clip(*self.clip) if self.clip else norm_batch


class Standardizer(Preprocessor):
    """Preprocessing module that performs image standardization."""

    def __init__(self, per_channel: bool = False, clip: Optional[Tuple[Union[float, int], Union[float, int]]] = None, **kwargs):
        """
        Args:
            per_channel (bool, optional): If True, standardization is done for each channel independently. Defaults to False.
            clip (Optional[Tuple[Union[float, int], Union[float, int]]], optional): If given, clip output to this range.
                Defaults to None.
        """
        super().__init__()
        self.per_channel = per_channel
        self.clip = clip

    def forward(self, batch: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        masked_batch, valid_mask = self.mask_batch_torch(batch, valid_mask, torch.nan)

        if self.per_channel:
            mean = masked_batch.nanmean(dim=(-2, -1))[:, :, None, None]
            std = torch.sqrt(torch.pow(masked_batch - mean, 2).nanmean(dim=(-2, -1)))[:, :, None, None]
            invalid = torch.zeros(1, device=batch.device)
        else:
            mean = masked_batch.nanmean(dim=(-3, -2, -1))[:, None, None, None]
            std = torch.sqrt(torch.pow(masked_batch - mean, 2).nanmean(dim=(-3, -2, -1)))[:, None, None, None]
            # invalid value is the channel mean, if available, else image mean for the unavailable channels
            invalid = masked_batch.nanmean(dim=(-2, -1))[:, :, None, None]
            invalid = torch.where(torch.isfinite(invalid), invalid, torch.nanmean(invalid))

        std_batch = torch.where(valid_mask, torch.divide(batch - mean, std), invalid)

        return std_batch.clip(*self.clip) if self.clip else std_batch


class HistogramEqualizer(Preprocessor):
    """Preprocessing module that performs histogram equalization."""

    def __init__(
        self,
        calc_framework: str = "torch",
        nbins: int = 512,
        target_domain: Tuple[Union[float, int], Union[float, int]] = (-1, 1),
        per_channel: bool = False,
        **kwargs,
    ):
        """
        Args:
            calc_framework (str): Calculation framework to use.
            nbins (int, optional): Number of bins for histogram equalization. Defaults to 512.
            target_domain (Tuple[Union[float, int], Union[float, int]], optional): Target domain for equalized image. Defaults to (-1, 1).
            per_channel (bool, optional): If True, histogram equalization is done for each channel independently. Defaults to False.
        """
        super().__init__()
        calc_framworks = ["numpy", "torch"]
        if calc_framework not in calc_framworks:
            raise NotImplementedError(f"Unknown calc_framework '{calc_framework}'. Choose one of {calc_framworks}.")
        super().__init__()
        self.calc_framework = calc_framework
        self.nbins = nbins
        self.min, self.max = target_domain
        self.per_channel = per_channel

    @staticmethod
    def _torch_interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """One-dimensional linear interpolation for monotonically increasing sample points.
        Returns the one-dimensional piecewise linear interpolant to a function with
        given discrete data points (xp, fp) evaluated at x.

        Args:
            x (torch.Tensor): x-coordinates at which to evaluate the interpolated values.
            xp (torch.Tensor): x-coordinates of the data points, must be increasing.
            fp (torch.Tensor): y-coordinates of the data points, same length as xp.

        Returns:
            torch.Tensor: Interpolated values, same size as x.
        """

        m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
        b = fp[:-1] - (m * xp[:-1])

        indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
        indicies = torch.clamp(indicies, 0, len(m) - 1)

        return m[indicies] * x + b[indicies]

    # workaround, because torch.histogram has no CUDA support, so far
    @staticmethod
    def _torch_histogram(x: torch.Tensor, nbins: int) -> Tuple[torch.Tensor, torch.Tensor]:
        min, max = x.min(), x.max()
        counts = torch.histc(x, nbins, min=min, max=max)
        edges = torch.linspace(min, max, nbins + 1, device=counts.device)

        return counts, edges

    def _torch_equalize_image(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None, nbins=100) -> torch.Tensor:
        if mask is None:
            mask = torch.all(torch.isfinite(image), dim=-3)
        counts, edges = self._torch_histogram(image[:, mask], nbins=nbins)
        bin_centers = edges[:-1] + torch.diff(edges) / 2.0
        cdf = counts.cumsum(dim=-1)
        cdf = cdf / cdf[-1]

        eq_image = self._torch_interp(image.flatten(), bin_centers, cdf).reshape(image.shape)
        return eq_image

    def _torch_forward(self, batch: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        valid_mask = self.combine_valid_masks_torch(batch, valid_mask)

        if self.per_channel:
            batch_eq = (
                torch.where(
                    valid_mask,
                    torch.stack(
                        [
                            torch.stack(
                                [self._torch_equalize_image(channel[None], mask[-1], self.nbins)[-1] for channel in image]
                            )
                            for image, mask in zip(batch, valid_mask)
                        ]
                    ),
                    torch.empty(1, device=batch.device).fill_(torch.nan),
                )
                * (self.max - self.min)
                + self.min
            )
            invalid = batch_eq.nanmean(dim=(-2, -1))[:, :, None, None]
            invalid = torch.where(torch.isfinite(invalid), invalid, (self.min + self.max) / 2)
        else:
            batch_eq = (
                torch.where(
                    valid_mask,
                    torch.stack(
                        [self._torch_equalize_image(image, mask[-1], self.nbins) for image, mask in zip(batch, valid_mask)]
                    ),
                    torch.empty(1, device=batch.device).fill_(torch.nan),
                )
                * (self.max - self.min)
                + self.min
            )
            # invalid = batch_eq.nanmean(dim=(-3, -2, -1))[:, None, None, None] # mask with image mean
            invalid = batch_eq.nanmean(dim=(-2, -1))[:, :, None, None]
            invalid = torch.where(torch.isfinite(invalid), invalid, invalid.nanmean())

        return torch.where(valid_mask, batch_eq, invalid)

    def _numpy_forward(self, batch: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> torch.Tensor:
        valid_mask = self.combine_valid_mask_numpy(batch=batch, valid_mask=valid_mask)

        if self.per_channel:
            batch_eq = (
                np.where(
                    valid_mask,
                    np.stack(
                        [
                            np.stack(
                                [
                                    equalize_hist(channel, nbins=self.nbins, mask=channel_mask)
                                    for channel, channel_mask in zip(image, mask)
                                ]
                            )
                            for image, mask in zip(batch, valid_mask)
                        ]
                    ),
                    np.empty(1) * np.nan,
                )
                * (self.max - self.min)
                + self.min
            )
            invalid = np.nanmean(batch_eq, axis=(-2, -1))[:, :, None, None]
            invalid = np.where(np.isfinite(invalid), invalid, (self.min + self.max) / 2)
        else:
            batch_eq = (
                np.where(
                    valid_mask,
                    np.stack([equalize_hist(image, nbins=self.nbins, mask=mask) for image, mask in zip(batch, valid_mask)]),
                    np.empty(1) * np.nan,
                )
                * (self.max - self.min)
                + self.min
            )
            # invalid = batch_eq.nanmean(dim=(-3, -2, -1))[:, None, None, None] # mask with image mean
            invalid = np.nanmean(batch_eq, axis=(-2, -1))[:, :, None, None]
            invalid = np.where(np.isfinite(invalid), invalid, np.nanmean(invalid))

        return torch.as_tensor(np.where(valid_mask, batch_eq, invalid), dtype=torch.float32)

    def forward(self, batch: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.calc_framework == "numpy":
            if batch.is_cuda:
                device = batch.device
                batch = batch.cpu()
                valid_mask = valid_mask.cpu()
                return self._numpy_forward(batch=batch.numpy(), valid_mask=valid_mask.numpy()).to(device, non_blocking=True)
            else:
                return self._numpy_forward(batch=batch.numpy(), valid_mask=valid_mask.numpy())
        else:
            return self._torch_forward(batch=batch, valid_mask=valid_mask)
