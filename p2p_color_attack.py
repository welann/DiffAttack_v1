"""
P2P-based color editing attack with a single classification loss.

This module implements the "main scheme" discussed in METHOD_IMPROVEMENTS.md:
we keep the overall latent-space DiffAttack framework, but:
    - Use Prompt-to-Prompt–style attention control to constrain edits
      to attribute tokens (e.g. color words) while preserving structure.
    - Optimize only a classification loss; no explicit attention or
      pixel-space regularizers are added to the objective.

The implementation is organized around a single entry point:

    p2p_color_attack(...)

which can be called from scripts or notebooks. The function follows the
same high-level stages as diff_latent_attack.py / diff_latent_attack_p2p.py:
    1) Classifier evaluation on the clean image.
    2) DDIM inversion to obtain a latent trajectory for the clean image.
    3) Short optimization of unconditional embeddings for better
       reconstruction quality (optional but kept for consistency).
    4) P2P-style attention control built from a base and an edited prompt.
    5) Latent optimization driven only by the classification loss.
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
from utils import view_images



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


@torch.enable_grad()
def p2p_color_attack(
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
    verbose: bool = True,
):
    """
    Run a P2P-style color editing attack driven only by classification loss.

    Args:
        model: StableDiffusionPipeline-like object with .vae, .unet,
            .tokenizer, .text_encoder, .scheduler.
        label: Ground-truth class label (numpy array, tensor, or int).
        base_prompt: Prompt describing the clean image content
            (e.g. "a flower in a garden").
        edit_prompt: Edited prompt, typically differing from base_prompt
            only on a color/attribute token (e.g. "a blue flower in a garden").
        num_inference_steps: Number of DDIM steps for inversion / sampling.
        guidance_scale: Guidance scale for classifier-free guidance.
        image: Clean PIL image to start from; if None, this function expects
            that the caller handles latent initialization externally.
        model_name: Name of surrogate classifier (see other_attacks.model_selection).
        save_path: Optional prefix for saving visualizations with view_images.
        res: Resolution used for classifier preprocessing (e.g. 224).
        start_step: DDIM step at which the latent attack starts.
        iterations: Number of outer optimization iterations on the latent.
        step_size: Step size for Adam optimizer on the latent.
        proj_radius: Optional L2 projection radius in latent space to keep
            the updated latent close to the inverted latent.
        dataset_name: Dataset identifier controlling classifier scaling.
        verbose: Whether to print progress bars and logs.

    Returns:
        adv_image: Adversarial image as uint8 numpy array (H, W, C).
        clean_acc: Clean accuracy of the classifier on the input image.
        adv_acc: Accuracy of the classifier on the adversarial image.
    """
    if image is None:
        raise ValueError("image must be provided for p2p_color_attack.")

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
    contexts = _build_context_per_step(
        model=model,
        uncond_embeddings_per_step=uncond_embeddings_per_step,
        prompts=prompts,
    )

    # 5) P2P-style attention controller:
    #    - base_prompt is treated as the class/content description.
    #    - edit_prompt can include additional attribute tokens (e.g. color).
    base_inds, edit_inds, _ = get_class_and_edit_indices(
        model.tokenizer, base_prompt, edit_prompt
    )

    controller = AttentionRefine(
        res=res,
        num_steps=len(model.scheduler.timesteps[1 + start_step - 1 :]),
        base_inds=base_inds,
        edit_inds=edit_inds,
        alpha_end=1.0,
    )
    register_attention_control(model, controller)

    # Attack latent initialisation: we start from the inverted latent at start_step.
    latent_clean = inversion_latents[start_step - 1].detach()
    latent_adv = latent_clean.clone().detach().requires_grad_(True)

    optimizer_latent = optim.Adam([latent_adv], lr=step_size)
    cross_entropy = torch.nn.CrossEntropyLoss()

    timesteps = model.scheduler.timesteps[1 + start_step - 1 :]

    iterator = range(iterations)
    if verbose:
        iterator = tqdm(iterator, desc="P2PColorAttack")

    for iteration_index in iterator:
        controller.reset()

        # Concatenate clean and adversarial latents to match controller.batch_size=2.
        latents = torch.cat([latent_clean, latent_adv], dim=0)

        # Run the remainder of the DDIM sampling with P2P attention control.
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts[step_index]
            latents = diffusion_step(
                model=model,
                latents=latents,
                context=context_step,
                t=timestep,
                guidance_scale=guidance_scale,
            )

        # Decode only the adversarial branch (index 1).
        decoded_adv = model.vae.decode(1 / 0.18215 * latents[1:])["sample"]
        classifier_input = _classifier_preprocess_decoded(decoded_adv)

        logits = classifier(classifier_input)
        if dataset_name != "imagenet_compatible":
            logits_for_loss = logits / 10
        else:
            logits_for_loss = logits

        attack_loss = -cross_entropy(logits_for_loss, label_tensor)

        optimizer_latent.zero_grad()
        attack_loss.backward()
        optimizer_latent.step()

        # Optional L2 projection to keep the adversarial latent close to the clean one.
        if proj_radius is not None:
            with torch.no_grad():
                diff = latent_adv - latent_clean
                diff_flat = diff.view(diff.shape[0], -1)
                norm = diff_flat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                factor = torch.clamp(proj_radius / norm, max=1.0)
                diff_flat = diff_flat * factor
                latent_adv.copy_((latent_clean.view(diff.shape[0], -1) + diff_flat).view_as(diff))

        if verbose:
            preds = torch.argmax(logits_for_loss.detach(), dim=1)
            success = (preds != label_tensor).float().mean().item()
            if isinstance(iterator, tqdm):
                iterator.set_postfix(
                    loss=float(attack_loss.item()), success_rate=success
                )

        # Early stopping if all examples are misclassified.
        with torch.no_grad():
            preds = torch.argmax(logits_for_loss, dim=1)
            if (preds != label_tensor).all():
                break

    # Final decode and evaluation.
    with torch.no_grad():
        controller.reset()
        latents = torch.cat([latent_clean, latent_adv], dim=0)
        for step_index, timestep in enumerate(timesteps):
            context_step = contexts[step_index]
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
                save_path=f"{save_path}_p2p_color_compare.png",
            )
            view_images(
                np.expand_dims(perturbed * 255.0, axis=0),
                show=False,
                save_path=f"{save_path}_p2p_color_adv.png",
            )

    reset_attention_control(model)
    return adv_image, clean_acc, adv_acc


__all__ = [
    "p2p_color_attack",
]

