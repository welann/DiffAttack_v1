"""
P2P-based color editing attack (v2) with fixed attention templates.

This implements the idea of replacing the variance/self-attention losses
in ``diff_latent_attack.py`` with *reference attention* losses:

- We first run a reference diffusion pass and record:
    - Cross-attention maps for the (class) tokens of the base prompt;
    - Self-attention maps aggregated over the network.
- During the latent-space attack, at every optimization iteration we:
    - Run the standard DDIM sampling starting from the current latent;
    - Decode the adversarial branch for the classifier and compute a
      classification loss;
    - Re-aggregate the current cross/self attention maps and penalize
      their deviation from the fixed reference templates.

In effect, the diffusion model is encouraged to keep its internal
attention patterns close to those of the reference run, while the
classifier is still being deceived. This typically stabilizes structure
and semantics (object, layout) while still allowing attribute changes
such as color.

The main entry point is:

    p2p_color_attack_v2(...)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch import optim
from tqdm import tqdm

import other_attacks
from attentionControl import AttentionStore
from utils import aggregate_attention, view_images



# ========== Shared helpers (adapted from p2p_color_attack.py) ==========


def _to_label_tensor(label: Union[np.ndarray, Tensor, int]) -> Tensor:
    """Convert label in {numpy, tensor, int} format to a CUDA long tensor."""
    if isinstance(label, np.ndarray):
        label_tensor = torch.from_numpy(label)
    elif isinstance(label, Tensor):
        label_tensor = label
    else:
        label_tensor = torch.tensor([int(label)], dtype=torch.long)
    label_tensor = label_tensor.long().view(-1).cuda()
    return label_tensor


def _classifier_preprocess_pil(image: Image.Image, res: int) -> Tensor:
    """
    Preprocess a PIL image for the classifier.

    This mirrors the preprocessing logic used in diff_latent_attack.py.
    """
    resized = image.resize((res, res), resample=Image.LANCZOS)
    array = np.float32(resized) / 255.0
    array = array[:, :, :3]
    array[:, :] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    array[:, :] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    array = array.transpose((2, 0, 1))
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


def _classifier_preprocess_decoded(decoded: Tensor) -> Tensor:
    """
    Preprocess decoded images from the diffusion model for the classifier.

    Args:
        decoded: Tensor in latent-decoder output space, shape [B, 3, H, W],
                 typically in [-1, 1].
    Returns:
        Tensor normalized for ImageNet-style classifiers, shape [B, 3, H, W].
    """
    # Map from [-1, 1] to [0, 1]
    images = (decoded / 2 + 0.5).clamp(0, 1)
    # Go to HWC for per-channel normalization, then back to CHW.
    images = images.permute(0, 2, 3, 1)
    mean = torch.as_tensor(
        [0.485, 0.456, 0.406], dtype=images.dtype, device=images.device
    )
    std = torch.as_tensor(
        [0.229, 0.224, 0.225], dtype=images.dtype, device=images.device
    )
    images = images.sub(mean).div(std)
    images = images.permute(0, 3, 1, 2)
    return images


def _optimize_uncond_embeddings(
    model,
    inversion_latents: Sequence[Tensor],
    start_step: int,
    base_prompt: str,
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    inner_iter_base: int = 10,
    verbose: bool = True,
) -> List[Tensor]:
    """
    Optimize unconditional embeddings for a better reconstruction of the
    inverted latent trajectory (Section 3.4 in the paper).

    This follows the same idea as diff_latent_attack.py / diff_latent_attack_p2p.py:
    we keep the prompt fixed to the base prompt and refine the unconditional
    embeddings so that the DDIM forward process tracks the inversion path.

    Returns:
        A list of unconditional embeddings, one per DDIM step used in the attack.
    """
    init_prompt = [base_prompt]
    batch_size = len(init_prompt)

    # Use the latent at the DDIM start step as initial point.
    latent_start = inversion_latents[start_step - 1]

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(
        uncond_input.input_ids.to(model.device)
    )[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    # Initialize latents for the reconstruction phase.
    latent_start, latents = init_latent(
        latent_start, model, height, width, batch_size
    )

    uncond_embeddings.requires_grad_(True)
    optimizer_uncond = optim.AdamW([uncond_embeddings], lr=1e-1)
    mse_loss = torch.nn.MSELoss()

    timesteps = model.scheduler.timesteps[1 + start_step - 1 :]
    all_uncond_emb: List[Tensor] = []

    iterator = timesteps
    if verbose:
        iterator = tqdm(timesteps, desc="Optimize_uncond_embed")

    for step_index, timestep in enumerate(iterator):
        # For later timesteps, allow slightly more inner iterations.
        num_inner = inner_iter_base + 2 * step_index
        for _ in range(num_inner):
            context = torch.cat([uncond_embeddings, text_embeddings])
            out_latents = diffusion_step(
                model=model,
                latents=latents,
                context=context,
                t=timestep,
                guidance_scale=guidance_scale,
            )
            optimizer_uncond.zero_grad()
            target = inversion_latents[start_step - 1 + step_index + 1]
            loss = mse_loss(out_latents, target)
            loss.backward()
            optimizer_uncond.step()

        with torch.no_grad():
            context = torch.cat([uncond_embeddings, text_embeddings])
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context,
                t=timestep,
                guidance_scale=guidance_scale,
            ).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    uncond_embeddings.requires_grad_(False)
    return all_uncond_emb


def _build_context_per_step(
    model,
    uncond_embeddings_per_step: Sequence[Tensor],
    prompts: Sequence[str],
) -> List[Tensor]:
    """
    Build encoder_hidden_states for each DDIM step during the attack.

    Shapes follow the pattern used in diff_latent_attack.py:
        - Let B = len(prompts) be the logical batch size.
        - For classifier-free guidance we replicate latents twice
          and concatenate unconditional + conditional embeddings:
              encoder_hidden_states: [2 * B, L, D]
    """
    batch_size = len(prompts)

    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    contexts: List[Tensor] = []
    for uncond_embeddings in uncond_embeddings_per_step:
        # uncond_embeddings: [1, L, D] -> repeat to match batch_size (B)
        uncond_rep = torch.cat([uncond_embeddings] * batch_size)
        # concat along batch dimension: [2 * B, L, D]
        context = torch.cat([uncond_rep, text_embeddings])
        contexts.append(context)
    return contexts


# ========== Reference attention templates ==========


def _compute_reference_attention_templates(
    model,
    base_prompt: str,
    edit_prompt: str,
    latent_clean: Tensor,
    uncond_embeddings_per_step: Sequence[Tensor],
    timesteps: Sequence[Tensor],
    res: int,
    res_cross: int,
    res_self: int,
) -> Tuple[Tensor, Tensor, List[int]]:
    """
    Run a *reference* diffusion pass to obtain fixed cross/self attention maps.

    The pass uses the same prompts and scheduler as the main attack but without
    modifying the latent; we only record the attention maps via an
    AttentionStore-like controller.

    Returns:
        cross_ref_label: cross-attention map for the base prompt tokens,
            shape [Hc, Wc, T_label], where T_label = len(true_label_tokens).
        self_ref: self-attention map aggregated at resolution res_self,
            shape [Hs, Ws, N], where N is the number of spatial positions.
        true_label_token_indices: indices (1..len(tokens)-2) inside the
            tokenizer encoding of base_prompt used as "true label" tokens.
    """
    prompts = [base_prompt, edit_prompt]

    # Tokens of the base prompt; 1..-2 are non-special tokens.
    tokens_base = model.tokenizer.encode(base_prompt)
    true_label_token_indices = list(range(1, len(tokens_base) - 1))

    # Build per-step contexts for the reference run (batch size = 2 prompts).
    contexts_ref = _build_context_per_step(
        model=model,
        uncond_embeddings_per_step=uncond_embeddings_per_step,
        prompts=prompts,
    )

    # Use a plain AttentionStore to record attention; no P2P modification.
    ref_store = AttentionStore(res)
    register_attention_control(model, ref_store)

    with torch.no_grad():
        # Two identical latents (clean) so that branch 0/1 share the same input.
        latents = torch.cat([latent_clean, latent_clean], dim=0)
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts_ref[step_index]
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context_step,
                t=timestep,
                guidance_scale=0.0,  # only for recording attention structure
            )

        # Cross-attention reference at coarse resolution (e.g., 7x7).
        cross_ref_all = aggregate_attention(
            prompts,
            ref_store,
            res_cross,
            ("up", "down"),
            True,
            select=1,
            is_cpu=False,
        )
        # Restrict to "true label" tokens from the base prompt.
        if len(true_label_token_indices) > 0:
            idx = torch.as_tensor(true_label_token_indices, device=cross_ref_all.device)
            cross_ref_label = cross_ref_all[:, :, idx]
        else:
            cross_ref_label = cross_ref_all

        # Self-attention reference at a slightly finer resolution (e.g., 14x14).
        self_ref = aggregate_attention(
            prompts,
            ref_store,
            res_self,
            ("up", "down"),
            False,
            select=1,
            is_cpu=False,
        )

    reset_attention_control(model)

    return cross_ref_label.detach(), self_ref.detach(), true_label_token_indices


# ========== Main attack (classification loss + reference attention losses) ==========


@torch.enable_grad()
def p2p_color_attack_v2(
    model,
    label: Union[np.ndarray, Tensor, int],
    base_prompt: str,
    edit_prompt: str,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    image: Optional[Image.Image] = None,
    model_name: str = "inception",
    save_path: Optional[str] = None,
    res: int = 224,
    start_step: int = 15,
    iterations: int = 30,
    step_size: float = 1e-2,
    proj_radius: Optional[float] = None,
    dataset_name: str = "imagenet_compatible",
    lambda_cross_ref: float = 1.0,
    lambda_self_ref: float = 1.0,
    verbose: bool = True,
):
    """
    Run a P2P-style color editing attack with *fixed attention templates*.

    Compared to ``p2p_color_attack``, this variant adds two regularizers:

      - Cross-attention reference loss:
            L_cross_ref = || A_cross_curr(label_tokens) - A_cross_ref(label_tokens) ||^2
      - Self-attention reference loss:
            L_self_ref = || A_self_curr - A_self_ref ||^2

    where the reference maps come from an initial diffusion run started from
    the clean inverted latent, and the current maps are recomputed at every
    latent optimization step.

    Args:
        model: StableDiffusionPipeline-like object with .vae, .unet,
            .tokenizer, .text_encoder, .scheduler.
        label: Ground-truth class label (numpy array, tensor, or int).
        base_prompt: Prompt describing the clean image content
            (e.g. "a red flower in a garden").
        edit_prompt: Edited prompt, typically differing from base_prompt
            only on a color/attribute token (e.g. "a blue flower in a garden").
        num_inference_steps: Number of DDIM steps for inversion / sampling.
        guidance_scale: Guidance scale for classifier-free guidance.
        image: Clean PIL image to start from.
        model_name: Name of surrogate classifier (see other_attacks.model_selection).
        save_path: Optional prefix for saving visualizations with view_images.
        res: Resolution used for classifier preprocessing (e.g. 224).
        start_step: DDIM step at which the latent attack starts.
        iterations: Number of outer optimization iterations on the latent.
        step_size: Step size for Adam optimizer on the latent.
        proj_radius: Optional L2 projection radius in latent space.
        dataset_name: Dataset identifier controlling classifier scaling.
        lambda_cross_ref: Weight for the cross-attention reference loss.
        lambda_self_ref: Weight for the self-attention reference loss.
        verbose: Whether to print progress bars and logs.

    Returns:
        adv_image: Adversarial image as uint8 numpy array (H, W, C).
        clean_acc: Clean accuracy of the classifier on the input image.
        adv_acc: Accuracy of the classifier on the adversarial image.
    """
    if image is None:
        raise ValueError("image must be provided for p2p_color_attack_v2.")

    label_tensor = _to_label_tensor(label)

    # Freeze diffusion model components; we only optimize the latent.
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    # 1) Clean evaluation of the classifier on the benign image.
    clean_tensor = _classifier_preprocess_pil(image, height).cuda()
    clean_logits = classifier(clean_tensor)
    if dataset_name != "imagenet_compatible":
        clean_logits_for_acc = clean_logits / 10
    else:
        clean_logits_for_acc = clean_logits

    clean_pred = torch.argmax(clean_logits_for_acc, dim=1)
    clean_acc = (
        (clean_pred == label_tensor).sum().item() / float(len(label_tensor))
    )
    if verbose:
        print(f"\nAccuracy on benign examples: {clean_acc * 100:.2f}%")

    # 2) DDIM inversion using the base prompt.
    latent_T, inversion_latents = ddim_reverse_sample(
        image=image,
        prompt=[base_prompt],
        model=model,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,
        res=height,
    )
    # Reverse to align with forward DDIM timesteps.
    inversion_latents = inversion_latents[::-1]

    # 3) Optimize unconditional embeddings for better reconstruction.
    uncond_embeddings_per_step = _optimize_uncond_embeddings(
        model=model,
        inversion_latents=inversion_latents,
        start_step=start_step,
        base_prompt=base_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        verbose=verbose,
    )

    # 4) Build contexts for attack using [base_prompt, edit_prompt].
    prompts = [base_prompt, edit_prompt]
    contexts_attack = _build_context_per_step(
        model=model,
        uncond_embeddings_per_step=uncond_embeddings_per_step,
        prompts=prompts,
    )

    # Timesteps used in both reference/attack passes.
    timesteps = model.scheduler.timesteps[1 + start_step - 1 :]

    # 5) Compute fixed attention templates from a reference pass.
    latent_clean = inversion_latents[start_step - 1].detach()
    res_cross = max(1, res // 32)
    res_self = max(1, res // 16)

    cross_ref_label, self_ref, true_label_token_indices = _compute_reference_attention_templates(
        model=model,
        base_prompt=base_prompt,
        edit_prompt=edit_prompt,
        latent_clean=latent_clean,
        uncond_embeddings_per_step=uncond_embeddings_per_step,
        timesteps=timesteps,
        res=res,
        res_cross=res_cross,
        res_self=res_self,
    )

    # 6) P2P-style attention controller for the attack pass:
    #    base_prompt is treated as the class/content description; edit_prompt
    #    may differ on attribute tokens (e.g., color).
    base_inds, edit_inds, _ = get_class_and_edit_indices(
        model.tokenizer, base_prompt, edit_prompt
    )

    attack_controller = AttentionRefine(
        res=res,
        num_steps=len(timesteps),
        base_inds=base_inds,
        edit_inds=edit_inds,
        alpha_end=1.0,
    )
    register_attention_control(model, attack_controller)

    # Attack latent initialization: start from the inverted latent at start_step.
    latent_adv = latent_clean.clone().detach().requires_grad_(True)

    optimizer_latent = optim.Adam([latent_adv], lr=step_size)
    cross_entropy = torch.nn.CrossEntropyLoss()

    iterator = range(iterations)
    if verbose:
        iterator = tqdm(iterator, desc="P2PColorAttackV2")

    for _ in iterator:
        attack_controller.reset()

        # Concatenate clean and adversarial latents to match batch_size=2.
        latents = torch.cat([latent_clean, latent_adv], dim=0)

        # Run the remainder of the DDIM sampling with attention control.
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts_attack[step_index]
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context_step,
                t=timestep,
                guidance_scale=guidance_scale,
            )

        # Decode only the adversarial branch (index 1) for classification.
        decoded_adv = model.vae.decode(1 / 0.18215 * latents[1:])["sample"]
        classifier_input = _classifier_preprocess_decoded(decoded_adv)

        logits = classifier(classifier_input)
        if dataset_name != "imagenet_compatible":
            logits_for_loss = logits / 10
        else:
            logits_for_loss = logits

        attack_loss = -cross_entropy(logits_for_loss, label_tensor)

        # Aggregate current attention maps.
        cross_curr_all = aggregate_attention(
            prompts,
            attack_controller,
            res_cross,
            ("up", "down"),
            True,
            select=1,
            is_cpu=False,
        )
        self_curr = aggregate_attention(
            prompts,
            attack_controller,
            res_self,
            ("up", "down"),
            False,
            select=1,
            is_cpu=False,
        )

        # Restrict cross-attention to "true label" tokens.
        if len(true_label_token_indices) > 0:
            idx = torch.as_tensor(
                true_label_token_indices, device=cross_curr_all.device
            )
            cross_curr_label = cross_curr_all[:, :, idx]
        else:
            cross_curr_label = cross_curr_all

        # Reference attention losses (MSE over maps).
        cross_ref_loss = torch.mean(
            (cross_curr_label - cross_ref_label.to(cross_curr_label.device)) ** 2
        )
        self_ref_loss = torch.mean(
            (self_curr - self_ref.to(self_curr.device)) ** 2
        )

        loss = attack_loss + lambda_cross_ref * cross_ref_loss + lambda_self_ref * self_ref_loss

        optimizer_latent.zero_grad()
        loss.backward()
        optimizer_latent.step()

        # Optional L2 projection to keep the adversarial latent close to the clean one.
        if proj_radius is not None:
            with torch.no_grad():
                diff = latent_adv - latent_clean
                diff_flat = diff.view(diff.shape[0], -1)
                norm = diff_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                factor = torch.clamp(proj_radius / norm, max=1.0)
                diff_flat = diff_flat * factor
                latent_adv.copy_(
                    (latent_clean.view(diff.shape[0], -1) + diff_flat).view_as(diff)
                )

        if verbose and isinstance(iterator, tqdm):
            preds = torch.argmax(logits_for_loss.detach(), dim=1)
            success = (preds != label_tensor).float().mean().item()
            iterator.set_postfix(
                loss=float(loss.item()),
                atk=float(attack_loss.item()),
                cross=float(cross_ref_loss.item()),
                self=float(self_ref_loss.item()),
                success_rate=success,
            )

        # Early stopping if all examples are misclassified.
        with torch.no_grad():
            preds = torch.argmax(logits_for_loss, dim=1)
            if (preds != label_tensor).all():
                break

    # Final decode and evaluation.
    with torch.no_grad():
        attack_controller.reset()
        latents = torch.cat([latent_clean, latent_adv], dim=0)
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts_attack[step_index]
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context_step,
                t=timestep,
                guidance_scale=guidance_scale,
            )

        decoded_adv = model.vae.decode(1 / 0.18215 * latents[1:])["sample"]
        classifier_input = _classifier_preprocess_decoded(decoded_adv)
        logits_adv = classifier(classifier_input)
        if dataset_name != "imagenet_compatible":
            logits_adv_for_acc = logits_adv / 10
        else:
            logits_adv_for_acc = logits_adv

        pred_adv = torch.argmax(logits_adv_for_acc, dim=1)
        adv_acc = (
            (pred_adv == label_tensor).sum().item() / float(len(label_tensor))
        )

        if verbose:
            print(f"Accuracy on adversarial examples: {adv_acc * 100:.2f}%")

        # Convert to numpy uint8 images for visualization/saving.
        adv_image = latent2image(model.vae, latents[1:].detach())[0]

        if save_path is not None:
            clean_image = latent2image(model.vae, latents[:1].detach())[0]
            real = clean_image.astype(np.float32) / 255.0
            perturbed = adv_image.astype(np.float32) / 255.0

            view_images(
                np.stack([real * 255.0, perturbed * 255.0], axis=0),
                show=False,
                save_path=f"{save_path}_p2p_color_v2_compare.png",
            )
            view_images(
                np.expand_dims(perturbed * 255.0, axis=0),
                show=False,
                save_path=f"{save_path}_p2p_color_v2_adv.png",
            )

    reset_attention_control(model)
    return adv_image, clean_acc, adv_acc


__all__ = [
    "p2p_color_attack_v2",
]

