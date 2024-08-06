from abc import ABC
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from skimage.exposure import equalize_hist


class BaseNormalizer(ABC):
    def combine_valid_masks(self, batch: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if valid_mask is None:
            valid_mask = np.all(np.isfinite(batch), axis=-3)
        elif not valid_mask.shape == batch[:, -1, :, :].shape:
            raise ValueError(f"valid_mask has wrong shape {valid_mask.shape}. Should be {batch[:,-1,:,:].shape}.")

        valid_mask = valid_mask[:, None, :, :] & np.isfinite(batch)

        return valid_mask

    def mask_batch(
        self, batch: np.ndarray, valid_mask: Optional[np.ndarray] = None, invalid_value: Optional[Any] = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid_mask = self.combine_valid_masks(batch=batch, valid_mask=valid_mask)
        masked_batch = np.where(valid_mask, batch, np.empty(1) * invalid_value)

        return masked_batch, valid_mask


class Normalizer(BaseNormalizer):
    def __init__(self, mean: List[float], std: List[float], clip: Optional[Tuple[Union[float, int], Union[float, int]]] = None):
        """Normalized image by (image - mean) / std with predifined std and mean.

        Args:
            mean (np.ndarray): Predifined mean for each channel [C].
            std (np.ndarray): Predifined std for each channel [C].
            clip (Option[Tuple[Union[float, int], Union[float, int]]], optional): If given, clip output to this range.
                Defaults to None.

        Returns:
            np.ndarray: Standardized image batch [B, C, H, W].
        """
        super().__init__()
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)
        self.clip = clip

    def __call__(self, batch: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Runs the image normalization.

        Invalid data (masked by valid mask) is set to NaN.

        Args:
            batch (np.ndarray): Input image batch [B, C, H, W].
            valid_mask (Optional[np.ndarray], optional): Mask for valid values (True) and NaN values (False) [B, H, W].
                Defaults to None.
        Raises:
            ValueError: If valid_mask has the wrong shape.

        Returns:
            np.ndarray: Normalized image batch [B, C, H, W].
        """
        masked_batch, valid_mask = self.mask_batch(batch=batch, valid_mask=valid_mask, invalid_value=0)

        norm_batch = np.divide(masked_batch - self.mean[None, :, None, None], self.std[None, :, None, None])

        return norm_batch.clip(*self.clip) if self.clip else norm_batch


class Standardizer(BaseNormalizer):
    def __init__(self, clip: Optional[Tuple[Union[float, int], Union[float, int]]] = None, per_channel: bool = False) -> None:
        """Standardize image by (image - mean) / std.

        NaN values (masked by valid mask) are assigned to the corresponding channel mean.

        Args:
            clip (Option[Tuple[Union[float, int], Union[float, int]]], optional): If given, clip output to this range.
                Defaults to None.
            per_channel (bool, optional): If True, standardization is done for each channel independently.
                Defaults to False.
        """
        super().__init__()
        self.clip = clip
        self.per_channel = per_channel

    def __call__(self, batch: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Runs the image normalization.

        NaN values (masked by valid mask) are assigned to the corresponding channel mean.

        Args:
            batch (np.ndarray): Input image batch [B, C, H, W].
            valid_mask (Optional[np.ndarray], optional): Mask for valid values (True) and NaN values (False) [B, H, W].
                Defaults to None.
        Raises:
            ValueError: If valid_mask has the wrong shape.

        Returns:
            np.ndarray: Normalized image batch [B, C, H, W].
        """
        masked_batch, valid_mask = self.mask_batch(batch=batch, valid_mask=valid_mask, invalid_value=np.nan)

        if self.per_channel:
            mean = np.nanmean(masked_batch, axis=(-2, -1))[:, :, None, None]
            std = np.nanstd(masked_batch, axis=(-2, -1))[:, :, None, None]
            invalid = np.zeros(1)
        else:
            mean = np.nanmean(masked_batch, axis=(-3, -2, -1))[:, None, None, None]
            std = np.nanstd(masked_batch, axis=(-3, -2, -1))[:, None, None, None]
            # invalid value is the channel mean, if available, else image mean for the unavailable channels
            invalid = np.nanmean(masked_batch, axis=(-2, -1))[:, :, None, None]
            invalid = np.where(np.isfinite(invalid), invalid, np.nanmean(invalid))

        std_batch = np.where(valid_mask, np.divide(batch - mean, std), invalid)

        return std_batch.clip(*self.clip) if self.clip else std_batch


class HistogramEqualizer(BaseNormalizer):
    def __init__(
        self,
        nbins: int = 1024,
        target_domain: Tuple[Union[float, int], Union[float, int]] = (-1, 1),
        per_channel: bool = False,
    ) -> None:
        """(Channel-wise) histogram equalization to the target domain.

        Args:
            nbins (int, optional): Number of bins for histogram equalization. Defaults to 1024.
            target_domain (Tuple[Union[float, int], Union[float, int]]): Target domain for equalized image.
                Defaults to (-1, 1).
            per_channel (bool, optional): If True, histogram equalization is done for each channel independently.
                Defaults to False.
        """
        super().__init__()
        self.nbins = nbins
        self.target_domain = target_domain
        self.per_channel = per_channel

    def __call__(self, batch: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Runs the image normalization.

        NaN values (masked by valid mask) are assigned to the corresponding channel mean.

        Args:
            batch (np.ndarray): Input image batch [B, C, H, W].
            valid_mask (Optional[np.ndarray], optional): Mask for valid values (True) and NaN values (False) [B, H, W].
                Defaults to None.
        Raises:
            ValueError: If valid_mask has the wrong shape.

        Returns:
            np.ndarray: Normalized image batch [B, C, H, W].
        """
        valid_mask = self.combine_valid_masks(batch, valid_mask)

        dmin, dmax = self.target_domain

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
                * (dmax - dmin)
                + dmin
            )
            invalid = np.nanmean(batch_eq, axis=(-2, -1))[:, :, None, None]
            invalid = np.where(np.isfinite(invalid), invalid, (dmin + dmax) / 2)
        else:
            batch_eq = (
                np.where(
                    valid_mask,
                    np.stack([equalize_hist(image, nbins=self.nbins, mask=mask) for image, mask in zip(batch, valid_mask)]),
                    np.empty(1) * np.nan,
                )
                * (dmax - dmin)
                + dmin
            )
            # invalid = batch_eq.nanmean(dim=(-3, -2, -1))[:, None, None, None] # mask with image mean
            invalid = np.nanmean(batch_eq, axis=(-2, -1))[:, :, None, None]
            invalid = np.where(np.isfinite(invalid), invalid, np.nanmean(invalid))

        return np.where(valid_mask, batch_eq, invalid)
