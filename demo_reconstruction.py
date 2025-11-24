from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from var_inv.measurements import (
    GaussianBlurMeasurement,
    MaskingMeasurement,
    SuperResolutionMeasurement,
)
from var_inv.posterior_sampling import (
    FrequencyAwareFilter,
    GradientGuidedVARSampler,
    MeasurementModel,
    PosteriorGuidanceConfig,
)
from var_inv.var_models.models.var import VAR
from var_inv.var_models.models.vqvae import VQVAE


def load_image(path: Path, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def save_image(tensor: torch.Tensor, path: Path, size: Tuple[int, int] | None = None) -> None:
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    img = tensor.squeeze(0).permute(1, 2, 0).numpy()
    img = (img * 255.0).astype(np.uint8)
    pil_img = Image.fromarray(img)
    if size is not None and pil_img.size != size:
        pil_img = pil_img.resize(size, Image.BICUBIC)
    pil_img.save(path)


def build_toy_var(device: torch.device) -> VAR:
    patch_nums = (1, 4, 16)
    vae = VQVAE(
        vocab_size=128,
        z_channels=16,
        ch=64,
        v_patch_nums=patch_nums,
        test_mode=True,
    ).to(device)
    var = VAR(
        vae_local=vae,
        num_classes=1,
        depth=2,
        embed_dim=128,
        num_heads=4,
        mlp_ratio=2.0,
        cond_drop_rate=0.0,
        patch_nums=patch_nums,
        flash_if_available=False,
        fused_if_available=False,
    ).to(device)
    return var


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo VAR posterior sampling with measurement guidance.")
    parser.add_argument("--image", type=Path, default=Path("diffusion-posterior-sampling/data/samples/00014.png"))
    parser.add_argument("--measurement", choices=("mask", "blur", "superres"), default="mask")
    parser.add_argument("--keep_ratio", type=float, default=0.4, help="Visible ratio for mask measurement.")
    parser.add_argument("--output_dir", type=Path, default=Path("var_inv/demo_outputs"))
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    var = build_toy_var(device)
    template = torch.zeros(1, var.Cvae, var.patch_nums[-1], var.patch_nums[-1], device=device)
    target_hw = var.vae_proxy[0].fhat_to_img(template).shape[-1]
    
    gt = load_image(args.image, target_hw)
    torch.manual_seed(0)
    
    if args.measurement == "mask":
        measurement = MaskingMeasurement.random(
            shape=(1, 1, gt.shape[-2], gt.shape[-1]),
            keep_ratio=args.keep_ratio,
            generator=torch.Generator().manual_seed(0),
        )
        measurement_model = MeasurementModel(operator=measurement.operator)
    elif args.measurement == "blur":
        measurement = GaussianBlurMeasurement(kernel_size=11, sigma=2.5, downsample=1)
        measurement_model = MeasurementModel(operator=measurement.operator)
    else:  # superres
        measurement = SuperResolutionMeasurement(scale=4)
        measurement_model = MeasurementModel(operator=measurement.operator, filter_fn=lambda x, _: x)
    
    measurement_tensor = measurement.measure(gt)
    
    config = PosteriorGuidanceConfig(
        cfg_scale=1.0,
        grad_scale_min=0.05,
        grad_scale_max=0.3,
        grad_start_ratio=0.0,
        top_p=0.9,
    )
    if measurement_model.filter_fn is None:
        measurement_model.filter_fn = FrequencyAwareFilter(config)
    
    sampler = GradientGuidedVARSampler(var_model=var, measurement_model=measurement_model, config=config)
    
    result = sampler.sample(measurement=measurement_tensor, label_B=0, g_seed=0)
    if isinstance(result, tuple):
        recon, _ = result
    else:
        recon = result
    
    mse = F.mse_loss(recon, gt).item()
    print(f"[demo] measurement={args.measurement}, mse_vs_gt={mse:.4f}")
    
    save_image(gt, args.output_dir / "ground_truth.png")
    meas_vis = measurement_tensor
    if args.measurement == "superres":
        meas_vis = F.interpolate(meas_vis, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    save_image(meas_vis, args.output_dir / f"{args.measurement}_measurement.png", size=(gt.shape[-1], gt.shape[-2]))
    save_image(recon, args.output_dir / f"{args.measurement}_reconstruction.png")
    print(f"[demo] Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
