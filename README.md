# VAR as Discrete Diffusion Posterior Sampling

This note records how to reinterpret a hierarchical Vector-Quantized AutoRegressive (VAR) generator as a discrete diffusion process and inject measurement consistency via gradient guidance, inspired by Diffusion Posterior Sampling (DPS).

## 1. Discrete Diffusion View
- Each VAR scale \(k\) predicts a discrete latent \(r_k\) conditioned on coarser scales \(r_{<k}\). Treat this as a Markov chain analogous to discrete diffusion, where the decoder maps \(r_k\) to an image estimate \(\hat{x}_k = \text{Decoder}(r_{\le k})\).
- The prior logits produced by the autoregressive transformer correspond to the diffusion prior; we can perturb them using gradients derived from the measurement likelihood.

## 2. Gradient-Guided Posterior Sampling
1. At step \(k\), obtain the logits over the codebook: \(\text{Logits}_{\text{orig}}\).
2. Construct a differentiable surrogate for sampling using Gumbel-Softmax or a straight-through estimator so that the logits influence a continuous relaxation \(r^{\text{soft}}_k\).
3. Decode \(r^{\text{soft}}_k\) (together with cached coarser latents) to get \(\hat{x}_k\), apply the measurement operator \(A\), and compute the loss
\[
\mathcal{L}_k = \| A(\hat{x}_k) - y \|_2^2.
\]
4. Backpropagate to obtain \(\nabla_{\text{logits}} \mathcal{L}_k\).
5. Update the logits before sampling:
\[
\text{Logits}_{\text{new}} = \text{Logits}_{\text{orig}} - \alpha_k \nabla_{\text{logits}} \mathcal{L}_k,
\]
   where \(\alpha_k\) can be scheduled per scale (often larger on coarse levels).
6. Sample \(r_k\) from the updated logits (or take argmax for deterministic decoding) and continue to the next scale.

> Implementation hint: cache the decoder activations so the additional backward pass over the logits is cheap; the gradient is only taken w.r.t. the logits, while the token sampling itself stays discrete via the straight-through estimator.

## 3. Approximation Quality Across Scales
- Coarse scales (e.g., \(1 \times 1, 2 \times 2\)) only capture low-frequency content; forcing them to match the full-resolution measurement creates large loss values and poor guidance.
- Either downsample \(y\) to the same spatial size before computing \(\mathcal{L}_k\), or low-pass filter both \(A(\hat{x}_k)\) and \(y\) so that the coarse logits only learn low-frequency corrections.
- Gradually increase the bandwidth or remove downsampling once the generated scale reaches the Nyquist frequency of the measurement.
- Optionally add a multi-term loss \(\lambda_k^\text{low} \| \text{LPF}(A(\hat{x}_k)) - \text{LPF}(y) \|^2 + \lambda_k^\text{high} \| A(\hat{x}_k) - y \|^2\) to smoothly transition between frequency bands.

## 4. Classifier-Free Guidance (CFG) Compatibility
- If the VAR already supports CFG, keep the usual conditional and unconditional logits, then apply both CFG and gradient guidance:
\[
\text{Logits}_{\text{final}} = \text{Logits}_{\text{cond}} + s_{\text{cfg}} (\text{Logits}_{\text{cond}} - \text{Logits}_{\text{uncond}}) - s_{\text{grad}} \nabla_{\text{logits}} \mathcal{L}_k.
\]
- \(s_{\text{cfg}}\) controls adherence to the textual/image prior, while \(s_{\text{grad}}\) tunes how aggressively the measurement residuals are enforced.
- Keep \(s_{\text{grad}}\) small on the earliest scales to avoid mode collapse, and anneal upward as higher-resolution scales start encoding high-frequency structures.

## 5. Practical Checklist
1. **Decoder access**: expose a function that maps relaxed tokens to pixel-space (or feature-space) images for every scale.
2. **Measurement model**: implement \(A(\cdot)\) and its differentiable approximation (e.g., masking, FFT subsampling, blur + subsample).
3. **Gradient hook**: wrap the logits with a custom autograd function implementing Gumbel-Softmax or straight-through so gradients flow even though the eventual token remains discrete.
4. **Loss scheduling**: define per-scale \(\alpha_k\), loss bandwidth, and gradient clip thresholds to avoid destabilizing the autoregressive chain.
5. **Sampling loop**: after applying CFG and gradient nudges, sample tokens, commit them to the context, and iterate over all scales.

With these components in place, VAR sampling becomes a form of discrete diffusion posterior sampling that respects both the learned prior and the measurement likelihood, bringing DPS-style reconstruction fidelity improvements to hierarchical discrete generators.

## 6. Reference Implementation

`posterior_sampling.py` contains a runnable implementation that wraps a pre-trained `VAR` checkpoint with gradient guidance:

```python
from var_inv.posterior_sampling import (
    GradientGuidedVARSampler,
    MeasurementModel,
    PosteriorGuidanceConfig,
)
from var_inv.var_models.models.var import VAR

var = ...  # load the pretrained checkpoint
config = PosteriorGuidanceConfig(cfg_scale=1.5, grad_scale_max=0.5)
sampler = GradientGuidedVARSampler(
    var_model=var,
    measurement_model=MeasurementModel(operator=lambda x: x),  # identity by default
    config=config,
)

# y_meas is A(x_true); measurement_model automatically applies a low-pass filter on coarse scales.
guided_imgs = sampler.sample(measurement=y_meas, label_B=class_indices)
```

The module exposes:
- `PosteriorGuidanceConfig`: handles CFG ratio, gradient scales, Gumbel-Softmax temperature, and low-pass scheduling.
- `FrequencyAwareFilter`: progressive blur/downsample utility applied to both predictions and observations to focus on low-frequency alignment at coarse scales.
- `MeasurementModel`: thin wrapper around the measurement operator \(A(\cdot)\) and optional filter.
- `GradientGuidedVARSampler`: drop-in replacement for `VAR.autoregressive_infer_cfg` that performs classifier-free guidance and gradient nudging at every scale via Gumbel-Softmax relaxations.

Hooking your own measurement is as simple as supplying a differentiable operator `A`. For example:

```python
mask = mask.to(device)
def A(x):
    return x * mask  # e.g., inpainting / compressed sensing mask

meas_model = MeasurementModel(operator=A)
sampler = GradientGuidedVARSampler(var, meas_model, PosteriorGuidanceConfig())
reconstruction = sampler.sample(measurement=y_meas)
```

This brings Diffusion Posterior Sampling-style measurement consistency into the hierarchical VAR decoder with only a few lines of code.

## 7. Measurement Library & Demo

`measurements.py` implements the most common operators \(A(\cdot)\) used in inverse problems:
- `IdentityMeasurement`: passthrough for baseline comparisons.
- `MaskingMeasurement`: binary masks for inpainting / compressed sensing (with `random(...)` helper).
- `GaussianBlurMeasurement`: blur kernels with optional downsampling.
- `SuperResolutionMeasurement`: area downsampling for low-resolution observations.

`demo_reconstruction.py` wires these measurements into the gradient-guided sampler using a lightweight toy VAR (depth=2, random weights). Running:

```bash
python var_inv/demo_reconstruction.py --measurement mask        # or blur / superres
```

will (1) load `diffusion-posterior-sampling/data/samples/00014.png`, (2) apply the selected measurement to obtain \(y = A(x)\), (3) run `GradientGuidedVARSampler` with the corresponding `MeasurementModel`, and (4) save the ground truth, measurement, and reconstruction to `var_inv/demo_outputs/`. The VAR is untrained, so reconstructions are not photorealistic, but the script verifies that the measurement operators, low-frequency filtering, and gradient guidance all execute end-to-end.

> Note: `var_inv/var_models/` vendors the original VAR Transformer + VQVAE code so this project is self-contained—no external checkout of the upstream VAR repo is required.

For a full notebook demo that downloads official checkpoints from Hugging Face and visualizes per-scale posterior sampling, open `var_inv/notebooks/hf_var_posterior_demo.ipynb`.
