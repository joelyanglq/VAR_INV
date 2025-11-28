from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from var_inv.var_models.models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from var_inv.var_models.models.var import VAR


Tensor = torch.Tensor


def _lerp(start: float, end: float, ratio: float) -> float:
    ratio = min(max(ratio, 0.0), 1.0)
    return start + (end - start) * ratio


@dataclass
class PosteriorGuidanceConfig:
    """Hyper-parameters controlling classifier-free and gradient guidance."""
    
    cfg_scale: float = 1.5
    top_k: int = 0
    top_p: float = 0.0
    more_smooth: bool = False
    grad_scale_min: float = 0.05
    grad_scale_max: float = 0.35
    grad_start_ratio: float = 0.05
    grad_clip: float = 2.0
    tau_start: float = 1.0
    tau_end: float = 0.25
    max_lowpass_kernel: int = 9
    min_lowpass_kernel: int = 1
    max_downsample: int = 4
    grad_steps: int = 1
    grad_stop_before_stage: int = 0
    
    def grad_scale(self, ratio: float) -> float:
        if ratio < self.grad_start_ratio:
            return 0.0
        return _lerp(self.grad_scale_min, self.grad_scale_max, ratio)
    
    def gumbel_tau(self, ratio: float) -> float:
        return _lerp(self.tau_start, self.tau_end, ratio)
    
    def lowpass_kernel(self, ratio: float) -> int:
        if self.max_lowpass_kernel <= 1:
            return 1
        k = int(round(_lerp(self.max_lowpass_kernel, self.min_lowpass_kernel, ratio)))
        k = max(1, min(self.max_lowpass_kernel, k))
        return k + 1 if k % 2 == 0 else k
    
    def downsample_factor(self, ratio: float) -> int:
        if self.max_downsample <= 1:
            return 1
        f = int(round(_lerp(self.max_downsample, 1.0, ratio)))
        return max(1, min(self.max_downsample, f))


class FrequencyAwareFilter:
    """Low-pass filter that progressively decreases its aggressiveness along the generation pyramid."""
    
    def __init__(self, cfg: PosteriorGuidanceConfig):
        self.cfg = cfg
    
    def __call__(self, tensor: Tensor, ratio: float) -> Tensor:
        if tensor.ndim < 4:
            return tensor
        kernel = self.cfg.lowpass_kernel(ratio)
        if kernel > 1:
            padding = kernel // 2
            weight = tensor.new_full((tensor.shape[1], 1, kernel, kernel), 1.0 / (kernel * kernel))
            tensor = F.conv2d(tensor, weight, bias=None, stride=1, padding=padding, groups=tensor.shape[1])
        factor = self.cfg.downsample_factor(ratio)
        if factor > 1:
            h, w = tensor.shape[-2:]
            pooled = F.avg_pool2d(tensor, kernel_size=factor, stride=factor, ceil_mode=False)
            tensor = F.interpolate(pooled, size=(h, w), mode="bilinear", align_corners=False)
        return tensor


class MeasurementModel:
    """
    Wraps the measurement operator A and optional frequency filter used inside the guidance loss.
    Observation tensors must already be in the measurement space (i.e., y = A(x_true)).
    """
    
    def __init__(
        self,
        operator: Optional[Callable[[Tensor], Tensor]] = None,
        filter_fn: Optional[Callable[[Tensor, float], Tensor]] = None,
    ):
        self.operator = operator or (lambda x: x)
        self.filter_fn = filter_fn
    
    def loss(self, recon: Tensor, measurement: Tensor, ratio: float) -> Tensor:
        pred = self.operator(recon)
        target = measurement
        pred = self._apply_filter(pred, ratio)
        target = self._apply_filter(target, ratio)
        return (pred - target).pow(2).sum()
    
    def measure(self, x: Tensor) -> Tensor:
        """Apply the measurement operator A(x) to obtain raw observations y."""
        return self.operator(x)
    
    def _apply_filter(self, tensor: Tensor, ratio: float) -> Tensor:
        if self.filter_fn is None:
            return tensor
        return self.filter_fn(tensor, ratio)


class GradientGuidedVARSampler:
    """
    Implements diffusion posterior sampling for VAR by nudging logits with gradients of
    || A(decoder(r_≤k)) - y ||^2 computed through a differentiable Gumbel-Softmax relaxation.
    """
    
    def __init__(
        self,
        var_model: VAR,
        measurement_model: MeasurementModel,
        config: PosteriorGuidanceConfig,
    ):
        self.var = var_model.eval()
        if measurement_model.filter_fn is None:
            measurement_model.filter_fn = FrequencyAwareFilter(config)
        self.measurement_model = measurement_model
        self.config = config
        self._last_grad_stats = {"norm": 0.0, "delta": 0.0}
    
    @torch.no_grad()
    def _prepare_tokens(
        self,
        B: int,
        label_B: Optional[Union[int, Tensor]],
        rng: Optional[torch.Generator],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        var = self.var
        if label_B is None:
            label_B = torch.multinomial(var.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=var.num_classes if label_B < 0 else label_B, device=var.lvl_1L.device)
        label_B = label_B.to(var.lvl_1L.device)
        sos = var.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=var.num_classes)), dim=0))
        lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, var.first_l, -1) + var.pos_start.expand(2 * B, var.first_l, -1) + lvl_pos[:, :var.first_l]
        return sos, lvl_pos, next_token_map
    
    def sample(
        self,
        measurement: Tensor,
        label_B: Optional[Union[int, Tensor]] = None,
        g_seed: Optional[int] = None,
        capture_intermediate: bool = False,
        capture_token_trace: bool = False,
        token_constraints: Optional[List[dict]] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, List[Tensor]],
        Tuple[Tensor, List[Tensor], List[dict]],
        Tuple[Tensor, List[dict]],
    ]:
        var = self.var
        device = var.lvl_1L.device
        measurement = measurement.to(device)
        B = measurement.shape[0]
        if g_seed is None:
            rng = None
        else:
            var.rng.manual_seed(g_seed)
            rng = var.rng
        
        with torch.no_grad():
            sos, lvl_pos, next_token_map = self._prepare_tokens(B, label_B, rng)
            cond_BD = sos
            f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])
            stage_imgs: Optional[List[Tensor]] = [] if capture_intermediate else None
            token_trace: Optional[List[dict]] = [] if capture_token_trace else None
            
            for b in var.blocks:
                b.attn.kv_caching(True)
            try:
                cur_L = 0
                for si, pn in enumerate(var.patch_nums):
                    ratio = si / max(var.num_stages_minus_1, 1)
                    cur_L += pn * pn
                    cond_BD_or_gss = var.shared_ada_lin(cond_BD)
                    x = next_token_map
                    for b in var.blocks:
                        x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                    logits_twice = var.get_logits(x, cond_BD)
                    
                    cfg_ratio = self.config.cfg_scale * ratio
                    cond_logits = logits_twice[:B]
                    uncond_logits = logits_twice[B:]
                    cfg_logits = (1 + cfg_ratio) * cond_logits - cfg_ratio * uncond_logits
                    guided_logits = self._apply_gradient_guidance(
                        base_logits=cfg_logits,
                        measurement=measurement,
                        f_hat=f_hat,
                        si=si,
                        pn=pn,
                        ratio=ratio,
                        rng=rng,
                        allow_grad=si >= self.config.grad_stop_before_stage,
                    )
                    
                    if token_constraints is not None and si < len(token_constraints) and token_constraints[si] is not None:
                        tc = token_constraints[si]
                        mask = tc["mask"].to(device)
                        idx_force = tc["idx"].to(device)
                        guided_logits = self._apply_token_constraint(guided_logits, mask=mask, idx_force=idx_force)
                    
                    if token_trace is not None:
                        mask_sum = int(mask.sum().item()) if token_constraints is not None and si < len(token_constraints) and token_constraints[si] is not None else 0
                        token_trace.append({
                            "stage": si,
                            "cfg_top": cfg_logits.argmax(dim=-1).detach().cpu(),
                            "guided_top": guided_logits.argmax(dim=-1).detach().cpu(),
                            "grad_norm": self._last_grad_stats["norm"],
                            "max_delta": self._last_grad_stats["delta"],
                            "forced_tokens": mask_sum,
                        })
                    idx_Bl = sample_with_top_k_top_p_(guided_logits, rng=rng, top_k=self.config.top_k, top_p=self.config.top_p, num_samples=1)[:, :, 0]
                    if not self.config.more_smooth:
                        h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
                    else:
                        gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                        h_BChw = gumbel_softmax_with_rng(guided_logits.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ var.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
                    h_BChw = h_BChw.transpose(1, 2).reshape(B, var.Cvae, pn, pn)
                    f_hat, next_token_map = var.vae_quant_proxy[0].get_next_autoregressive_input(si, len(var.patch_nums), f_hat, h_BChw)
                    if stage_imgs is not None:
                        stage_imgs.append(self._decode_current_fhat(f_hat).detach().cpu())
                    if si != var.num_stages_minus_1:
                        next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
                        next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var.patch_nums[si + 1] ** 2]
                        next_token_map = next_token_map.repeat(2, 1, 1)
            finally:
                for b in var.blocks:
                    b.attn.kv_caching(False)
            
            img = self._decode_current_fhat(f_hat)
            if stage_imgs is not None:
                stage_imgs.append(img.detach().cpu())
                if token_trace is not None:
                    return img, stage_imgs, token_trace
                return img, stage_imgs
            if token_trace is not None:
                return img, token_trace
        return img
    
    def _apply_gradient_guidance(
        self,
        base_logits: Tensor,
        measurement: Tensor,
        f_hat: Tensor,
        si: int,
        pn: int,
        ratio: float,
        rng: Optional[torch.Generator],
        allow_grad: bool = True,
    ) -> Tensor:
        grad_scale = self.config.grad_scale(ratio)
        if not allow_grad:
            grad_scale = 0.0
        if grad_scale <= 0.0:
            return base_logits

        steps = max(1, self.config.grad_steps)
        guided_logits = base_logits.clone()
        max_delta = 0.0
        grad_norm = 0.0
        quant = self.var.vae_quant_proxy[0]
        SN = len(self.var.patch_nums)
        HW = quant.v_patch_nums[-1]
        for _ in range(steps):
            with torch.enable_grad():
                logits_for_grad = guided_logits.detach().clone()
                # 强制拉回 Logits 的分布，确保 Softmax 有梯度
                logits_std = logits_for_grad.std(dim=-1, keepdim=True) + 1e-6
                logits_for_grad = logits_for_grad / logits_std 

                logits_for_grad.requires_grad_(True)
                tau = max(self.config.gumbel_tau(ratio), 1e-3)
                soft_codes = gumbel_softmax_with_rng(logits_for_grad, tau=tau, hard=False, dim=-1, rng=rng)
                emb = quant.embedding.weight   # (V, Cvae)
                h_soft = soft_codes @ emb.unsqueeze(0)  # (B, L, Cvae)
                h_soft = h_soft.transpose(1, 2).reshape(base_logits.shape[0], self.var.Cvae, pn, pn)  # (B, Cvae, pn, pn)
                
                # 4. 上采样到全尺寸（若非最后一阶段）& 残差量化
                ratio_si = si / max(1, SN - 1)
                if si != SN - 1:
                    h_up = quant.quant_resi[ratio_si](F.interpolate(h_soft, size=(HW, HW), mode='bicubic'))
                else:
                    h_up = quant.quant_resi[ratio_si](h_soft)
                probe = f_hat.detach() + h_up  # 累积：前阶段 + 当前
                vae = self.var.vae_proxy[0]
                recon = vae.decoder(vae.post_quant_conv(probe)).add_(1).mul_(0.5)  # avoid clamp to keep gradients alive
                loss = self.measurement_model.loss(recon, measurement, ratio)
                grad = torch.autograd.grad(loss, logits_for_grad, retain_graph=False, create_graph=False)[0]

                g_std = grad.std()
                g_mean = grad.mean()
                
                # 避免除以零
                if g_std > 1e-8:
                     # 将梯度标准化到 unit variance
                    grad_normalized = (grad - g_mean) / g_std
                else:
                    grad_normalized = grad

                grad_norm = grad.abs().mean().item()
            
            if self.config.grad_clip > 0:
                grad = grad.clamp_(min=-self.config.grad_clip, max=self.config.grad_clip)
            update = grad_scale * grad
            guided_logits = guided_logits - update
            max_delta = max(max_delta, update.abs().max().item())
        
        self._last_grad_stats["norm"] = grad_norm
        self._last_grad_stats["delta"] = max_delta
        return guided_logits

    @staticmethod
    def _apply_token_constraint(logits: Tensor, mask: Tensor, idx_force: Tensor) -> Tensor:
        """
        Force logits at masked positions to take the given token index.
        logits: (B, l, V)
        mask:   (B, l) boolean
        idx_force: (B, l) long
        """
        if mask is None or mask.numel() == 0:
            return logits
        logits = logits.clone()
        mask_exp = mask.unsqueeze(-1)
        logits.masked_fill_(mask_exp, -torch.inf)
        logits.scatter_(2, idx_force.unsqueeze(-1), 1e4)
        return logits

    def _decode_current_fhat(self, f_hat: Tensor) -> Tensor:
        img = self.var.vae_proxy[0].fhat_to_img(f_hat.detach().clone()).add_(1).mul_(0.5)
        return img
