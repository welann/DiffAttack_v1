"""
Text-guided, semantically controllable adversarial attack in latent space.

This module extends the latent-space DiffAttack framework with:
    - A text prompt that explicitly encodes the desired semantic change
      (e.g. "background is a dark forest", "with tiger-like stripes").
    - A cross-attention–derived region mask that localizes the edited
      attribute tokens to specific image regions.
    - A latent optimization objective that combines classification loss
      with a content-preservation term applied outside the edited region.

High-level pipeline:
    1) Evaluate a surrogate classifier on the clean image.
    2) Run DDIM inversion with a base prompt describing the clean content.
    3) Optionally refine unconditional embeddings for better reconstruction.
    4) Build a semantic edit prompt:
           edit_prompt = base_prompt + " " + attribute_prompt
       and compute a cross-attention mask for the attribute tokens.
    5) Optimize the latent code to fool the classifier while preserving
       non-edited areas in pixel space, using the attention mask.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch import optim
from tqdm import tqdm

import other_attacks
from attentionControl import AttentionStore
from diff_latent_attack import (
    ddim_reverse_sample,
    diffusion_step,
    init_latent,
    latent2image,
    preprocess,
    register_attention_control,
    reset_attention_control,
)
from utils import aggregate_attention, view_images


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

    This follows the same idea as diff_latent_attack.py:
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


def _get_attribute_token_indices(
    tokenizer,
    edit_prompt: str,
    attribute_prompt: str,
) -> np.ndarray:
    """
    Map the attribute phrase to token indices inside the edit prompt.

    The implementation mirrors the word-to-token alignment used in
    Prompt-to-Prompt: we first locate word positions of the attribute
    phrase, then map them to subword token positions.
    """
    if attribute_prompt.strip() == "":
        # Fallback: all non-special tokens.
        tokens = tokenizer.encode(edit_prompt)
        return np.arange(1, len(tokens) - 1, dtype=np.int64)

    split_text = edit_prompt.split(" ")
    attr_words = [w for w in attribute_prompt.split(" ") if w]

    # Word-level positions where attribute words appear.
    word_positions = []
    for attr_word in attr_words:
        for idx, word in enumerate(split_text):
            if word.strip(",.") == attr_word:
                word_positions.append(idx)

    if not word_positions:
        tokens = tokenizer.encode(edit_prompt)
        return np.arange(1, len(tokens) - 1, dtype=np.int64)

    word_positions = sorted(set(word_positions))

    # Map word positions to token indices (1..len(tokens)-2).
    words_encode = [
        tokenizer.decode([item]).strip("#")
        for item in tokenizer.encode(edit_prompt)
    ][1:-1]

    out: List[int] = []
    cur_len = 0
    ptr = 0
    for i, token_str in enumerate(words_encode):
        if ptr >= len(split_text):
            break
        cur_len += len(token_str)
        if ptr in word_positions:
            out.append(i + 1)
        if cur_len >= len(split_text[ptr]):
            ptr += 1
            cur_len = 0

    if not out:
        tokens = tokenizer.encode(edit_prompt)
        return np.arange(1, len(tokens) - 1, dtype=np.int64)

    return np.asarray(sorted(set(out)), dtype=np.int64)


def _compute_attribute_region_mask(
    model,
    edit_prompt: str,
    attribute_prompt: str,
    latent_clean: Tensor,
    contexts_per_step: Sequence[Tensor],
    timesteps: Sequence[Tensor],
    res: int,
    res_cross: int,
    region_threshold: float,
    use_soft_mask: bool,
) -> Tensor:
    """
    Compute a spatial mask highlighting regions associated with the
    attribute tokens in the edit prompt, using cross-attention maps.

    Returns:
        Tensor of shape [1, 1, res, res] on model.device, values in [0, 1].
    """
    controller = AttentionStore(res=res)
    register_attention_control(model, controller)

    with torch.no_grad():
        controller.reset()
        # Single-branch latent; AttentionStore only records the conditional half.
        latents = latent_clean.clone()
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts_per_step[step_index]
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context_step,
                t=timestep,
                guidance_scale=model.scheduler.config.guidance_scale
                if hasattr(model.scheduler.config, "guidance_scale")
                else 0.0,
            )

        prompts = [edit_prompt]
        cross_maps = aggregate_attention(
            prompts,
            controller,
            res_cross,
            ("up", "down"),
            True,
            select=0,
            is_cpu=False,
        )

    reset_attention_control(model)

    # cross_maps: [Hc, Wc, T]
    attr_token_indices = _get_attribute_token_indices(
        model.tokenizer, edit_prompt, attribute_prompt
    )
    if attr_token_indices.size > 0:
        idx = torch.as_tensor(
            attr_token_indices,
            device=cross_maps.device,
            dtype=torch.long,
        )
        idx = idx.clamp(0, cross_maps.shape[-1] - 1)
        attr_maps = cross_maps[:, :, idx]
        attr_map = attr_maps.mean(-1)
    else:
        attr_map = cross_maps.mean(-1)

    # Normalize to [0, 1].
    attr_map = attr_map - attr_map.min()
    attr_map = attr_map / (attr_map.max() + 1e-8)

    # Upsample to image resolution.
    attr_map = attr_map.unsqueeze(0).unsqueeze(0)
    region_mask = F.interpolate(
        attr_map,
        size=(res, res),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)

    if not use_soft_mask:
        region_mask = (region_mask >= region_threshold).float()

    return region_mask.to(model.device)


@torch.enable_grad()
def p2p_color_attack_v3(
    model,
    label: Union[np.ndarray, Tensor, int],
    base_prompt: str,
    attribute_prompt: str,
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
    lambda_preserve: float = 1.0,
    region_threshold: float = 0.5,
    use_soft_mask: bool = False,
    region_mask: Optional[Tensor] = None,
    verbose: bool = True,
):
    """
    Text-guided, semantically controllable adversarial attack.

    Args:
        model: StableDiffusionPipeline-like object with .vae, .unet,
            .tokenizer, .text_encoder, .scheduler.
        label: Ground-truth class label (numpy array, tensor, or int).
        base_prompt: Clean semantic description of the image
            (e.g. "a photo of a dog").
        attribute_prompt: Attribute / edit description appended to the base
            prompt (e.g. "with tiger-like stripes", "background is a forest").
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
        lambda_preserve: Weight for the content-preservation loss outside
            the edited region.
        region_threshold: Threshold applied to the attention map when
            constructing a binary region mask (if use_soft_mask is False).
        use_soft_mask: If True, use a soft [0, 1] attention map instead
            of thresholding to a binary mask.
        region_mask: Optional pre-defined region mask tensor of shape
            [1, 1, res, res] on CUDA; if provided, attention-based mask
            computation is skipped.
        verbose: Whether to print progress bars and logs.

    Returns:
        adv_image: Adversarial image as uint8 numpy array (H, W, C).
        clean_acc: Clean accuracy of the classifier on the input image.
        adv_acc: Accuracy of the classifier on the adversarial image.
    """
    if image is None:
        raise ValueError("image must be provided for p2p_color_attack_v3.")

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

    # 4) Build edit prompt and per-step contexts.
    attribute_prompt = attribute_prompt.strip()
    if attribute_prompt:
        edit_prompt = f"{base_prompt.strip()} {attribute_prompt}"
    else:
        edit_prompt = base_prompt

    prompts_edit = [edit_prompt]
    contexts_edit = _build_context_per_step(
        model=model,
        uncond_embeddings_per_step=uncond_embeddings_per_step,
        prompts=prompts_edit,
    )

    timesteps = model.scheduler.timesteps[1 + start_step - 1 :]

    # 5) Attention-based region mask for the attribute.
    #    If a mask is provided, respect it; otherwise compute from cross-attention.
    if region_mask is not None:
        region_mask = region_mask.to(model.device)
        if region_mask.ndim != 4 or region_mask.shape[0] != 1:
            raise ValueError(
                "region_mask must have shape [1, 1, H, W]; "
                f"got {tuple(region_mask.shape)}"
            )
    elif lambda_preserve > 0.0:
        latent_clean = inversion_latents[start_step - 1].detach()
        res_cross = max(1, res // 32)
        region_mask = _compute_attribute_region_mask(
            model=model,
            edit_prompt=edit_prompt,
            attribute_prompt=attribute_prompt,
            latent_clean=latent_clean,
            contexts_per_step=contexts_edit,
            timesteps=timesteps,
            res=res,
            res_cross=res_cross,
            region_threshold=region_threshold,
            use_soft_mask=use_soft_mask,
        )
    else:
        region_mask = None

    # Original image in diffusion space for content preservation.
    orig_image_tensor = preprocess(image, res=height).to(model.device)

    # Attack latent initialisation: start from the inverted latent at start_step.
    latent_adv = inversion_latents[start_step - 1].detach().clone()
    latent_adv.requires_grad_(True)

    optimizer_latent = optim.Adam([latent_adv], lr=step_size)
    cross_entropy = torch.nn.CrossEntropyLoss()

    iterator = range(iterations)
    if verbose:
        iterator = tqdm(iterator, desc="P2PColorAttackV3")

    for _ in iterator:
        # Run the remainder of the DDIM sampling from the current latent.
        latents = latent_adv
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts_edit[step_index]
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context_step,
                t=timestep,
                guidance_scale=guidance_scale,
            )

        # Decode adversarial sample.
        decoded_adv = model.vae.decode(1 / 0.18215 * latents)["sample"]
        classifier_input = _classifier_preprocess_decoded(decoded_adv)

        logits = classifier(classifier_input)
        if dataset_name != "imagenet_compatible":
            logits_for_loss = logits / 10
        else:
            logits_for_loss = logits

        attack_loss = -cross_entropy(logits_for_loss, label_tensor)

        # Content-preservation loss outside the edited region.
        if lambda_preserve > 0.0 and region_mask is not None:
            preserve_mask = 1.0 - region_mask
            preserve_loss = F.mse_loss(
                decoded_adv * preserve_mask,
                orig_image_tensor * preserve_mask,
            )
        else:
            preserve_loss = torch.zeros(1, device=latent_adv.device)

        loss = attack_loss + lambda_preserve * preserve_loss

        optimizer_latent.zero_grad()
        loss.backward()
        optimizer_latent.step()

        # Optional L2 projection to keep the adversarial latent close to the inverted one.
        if proj_radius is not None:
            with torch.no_grad():
                diff = latent_adv - inversion_latents[start_step - 1]
                diff_flat = diff.view(diff.shape[0], -1)
                norm = diff_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                factor = torch.clamp(proj_radius / norm, max=1.0)
                diff_flat = diff_flat * factor
                latent_adv.copy_(
                    (
                        inversion_latents[start_step - 1]
                        .view(diff.shape[0], -1)
                        .add(diff_flat)
                    ).view_as(diff)
                )

        if verbose and isinstance(iterator, tqdm):
            preds = torch.argmax(logits_for_loss.detach(), dim=1)
            success = (preds != label_tensor).float().mean().item()
            iterator.set_postfix(
                loss=float(loss.item()),
                atk=float(attack_loss.item()),
                preserve=float(preserve_loss.item()),
                success_rate=success,
            )

        # Early stopping if all examples are misclassified.
        with torch.no_grad():
            preds = torch.argmax(logits_for_loss, dim=1)
            if (preds != label_tensor).all():
                break

    # Final decode and evaluation.
    with torch.no_grad():
        latents = latent_adv.detach()
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts_edit[step_index]
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context_step,
                t=timestep,
                guidance_scale=guidance_scale,
            )

        decoded_adv = model.vae.decode(1 / 0.18215 * latents)["sample"]
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
        adv_image = latent2image(model.vae, latents.detach())[0]

        if save_path is not None:
            # For comparison, reconstruct the clean image from inversion.
            clean_latents = inversion_latents[start_step - 1].detach()
            clean_latents, clean_latents_batch = init_latent(
                clean_latents, model, height, width, batch_size=1
            )
            model.scheduler.set_timesteps(num_inference_steps)
            for t in model.scheduler.timesteps[1 + start_step - 1 :]:
                clean_latents_batch = diffusion_step(
                    model=model,
                    latents=clean_latents_batch,
                    context=contexts_edit[0],
                    t=t,
                    guidance_scale=guidance_scale,
                )
            clean_image = latent2image(model.vae, clean_latents_batch.detach())[0]

            real = clean_image.astype(np.float32) / 255.0
            perturbed = adv_image.astype(np.float32) / 255.0

            view_images(
                np.stack([real * 255.0, perturbed * 255.0], axis=0),
                show=False,
                save_path=f"{save_path}_p2p_color_v3_compare.png",
            )
            view_images(
                np.expand_dims(perturbed * 255.0, axis=0),
                show=False,
                save_path=f"{save_path}_p2p_color_v3_adv.png",
            )

    return adv_image, clean_acc, adv_acc


__all__ = [
    "p2p_color_attack_v3",
]

