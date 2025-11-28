from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


class BaseMeasurement:
    """Base class for measurement operators A(x)."""
    
    def operator(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def measure(self, x: Tensor) -> Tensor:
        return self.operator(x)
    
    def to(self, device: torch.device) -> "BaseMeasurement":
        return self


class IdentityMeasurement(BaseMeasurement):
    def operator(self, x: Tensor) -> Tensor:
        return x


@dataclass
class MaskingMeasurement(BaseMeasurement):
    """A(x) = M * x, commonly used for inpainting or compressed sensing masks."""
    
    mask: Tensor  # shape (1 or B, 1 or C, H, W)
    
    def operator(self, x: Tensor) -> Tensor:
        return x * self.mask
    
    def to(self, device: torch.device) -> "MaskingMeasurement":
        self.mask = self.mask.to(device)
        return self
    
    @staticmethod
    def random(
        shape: torch.Size,
        keep_ratio: float = 0.5,
        generator: Optional[torch.Generator] = None,
    ) -> "MaskingMeasurement":
        mask = torch.rand(shape, generator=generator)
        mask = (mask < keep_ratio).float()
        return MaskingMeasurement(mask=mask)
    
    @staticmethod
    def rectangular(
        h: int,
        w: int,
        top: int,
        left: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> "MaskingMeasurement":
        """
        Build a rectangle mask where the rectangle region is set to 0 (unknown) and others to 1 (known).
        """
        mask = torch.ones(1, 1, h, w, device=device)
        mask[..., top:top+height, left:left+width] = 0.0
        return MaskingMeasurement(mask=mask)


@dataclass
class GaussianBlurMeasurement(BaseMeasurement):
    """A(x) = G_sigma * x, optionally followed by downsampling."""
    
    kernel_size: int = 9
    sigma: float = 3.0
    downsample: int = 1
    
    def operator(self, x: Tensor) -> Tensor:
        kernel = self._create_kernel(x.device, x.dtype)
        padding = self.kernel_size // 2
        c = x.shape[1]
        weight = kernel.expand(c, 1, -1, -1)
        blurred = F.conv2d(x, weight, padding=padding, groups=c)
        if self.downsample > 1:
            blurred = F.avg_pool2d(blurred, kernel_size=self.downsample, stride=self.downsample)
        return blurred
    
    def _create_kernel(self, device, dtype):
        ax = torch.arange(-(self.kernel_size // 2), self.kernel_size // 2 + 1, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@dataclass
class SuperResolutionMeasurement(BaseMeasurement):
    """A(x) downsamples by a given factor (area pooling)."""
    
    scale: int = 4
    
    def operator(self, x: Tensor) -> Tensor:
        if self.scale <= 1:
            return x
        return F.interpolate(x, scale_factor=1 / self.scale, mode="area")
