from typing import Optional, List, Tuple
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images, aggregate_attention
from distances import LpDistance
import other_attacks
from attentionControl import AttentionStore
import torch.nn.functional as F


# ========== Basic helpers (copied/adapted from diff_latent_attack.py) ==========


def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)  # type: ignore
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.no_grad()
def ddim_reverse_sample(
    image,
    prompt,
    model,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    res=512,
):
    """DDIM inversion identical to original implementation (single prompt)."""
    batch_size = 1

    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        prompt[0],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)

    model.scheduler.set_timesteps(num_inference_steps)

    latents = encoder(image, model, res=res)
    timesteps = model.scheduler.timesteps.flip(0)

    all_latents = [latents]

    for t in tqdm(timesteps[:-1], desc="DDIM_inverse"):
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]

        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )

        next_timestep = (
            t
            + model.scheduler.config.num_train_timesteps
            // model.scheduler.num_inference_steps
        )
        alpha_bar_next = (
            model.scheduler.alphas_cumprod[next_timestep]
            if next_timestep <= model.scheduler.config.num_train_timesteps
            else torch.tensor(0.0)
        )

        reverse_x0 = (
            1
            / torch.sqrt(model.scheduler.alphas_cumprod[t])
            * (latents - noise_pred * torch.sqrt(1 - model.scheduler.alphas_cumprod[t]))
        )
        latents = (
            reverse_x0 * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * noise_pred
        )
        all_latents.append(latents)

    return latents, all_latents


# def register_attention_control(model, controller):
#     def ca_forward(self, place_in_unet):
#         def forward(
#             hidden_states: torch.FloatTensor,
#             encoder_hidden_states: Optional[torch.FloatTensor] = None,
#             attention_mask: Optional[torch.FloatTensor] = None,
#             temb: Optional[torch.FloatTensor] = None,
#         ):
#             if self.spatial_norm is not None:
#                 hidden_states = self.spatial_norm(hidden_states, temb)

#             batch_size, sequence_length, _ = (
#                 hidden_states.shape
#                 if encoder_hidden_states is None
#                 else encoder_hidden_states.shape
#             )

#             if attention_mask is not None:
#                 attention_mask = self.prepare_attention_mask(
#                     attention_mask, sequence_length, batch_size
#                 )
#                 attention_mask = attention_mask.view(
#                     batch_size, self.heads, -1, attention_mask.shape[-1]
#                 )

#             if self.group_norm is not None:
#                 hidden_states = self.group_norm(
#                     hidden_states.transpose(1, 2)
#                 ).transpose(1, 2)

#             query = self.to_q(hidden_states)

#             is_cross = encoder_hidden_states is not None
#             if encoder_hidden_states is None:
#                 encoder_hidden_states = hidden_states
#             elif self.norm_cross:
#                 encoder_hidden_states = self.norm_encoder_hidden_states(
#                     encoder_hidden_states
#                 )
#             key = self.to_k(encoder_hidden_states)
#             value = self.to_v(encoder_hidden_states)

#             def reshape_heads_to_batch_dim(tensor):
#                 b, seq_len, dim = tensor.shape
#                 h = self.heads
#                 tensor = tensor.reshape(b, seq_len, h, dim // h)
#                 tensor = tensor.permute(0, 2, 1, 3).reshape(b * h, seq_len, dim // h)
#                 return tensor

#             query = reshape_heads_to_batch_dim(query)
#             key = reshape_heads_to_batch_dim(key)
#             value = reshape_heads_to_batch_dim(value)

#             sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
#             attn = sim.softmax(dim=-1)
#             attn = controller(attn, is_cross, place_in_unet)
#             out = torch.einsum("b i j, b j d -> b i d", attn, value)

#             def reshape_batch_dim_to_heads(tensor):
#                 b, seq_len, dim = tensor.shape
#                 h = self.heads
#                 tensor = tensor.reshape(b // h, h, seq_len, dim)
#                 tensor = tensor.permute(0, 2, 1, 3).reshape(b // h, seq_len, dim * h)
#                 return tensor

#             out = reshape_batch_dim_to_heads(out)
#             out = self.to_out[0](out)
#             out = self.to_out[1](out)
#             out = out / self.rescale_output_factor
#             return out

#         return forward

#     def register_recr(net_, count, place_in_unet):
#         if net_.__class__.__name__ == "Attention":
#             net_.forward = ca_forward(net_, place_in_unet)
#             return count + 1
#         elif hasattr(net_, "children"):
#             for net__ in net_.children():
#                 count = register_recr(net__, count, place_in_unet)
#         return count

#     cross_att_count = 0
#     sub_nets = model.unet.named_children()
#     for net in sub_nets:
#         if "down" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "down")
#         elif "up" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "up")
#         elif "mid" in net[0]:
#             cross_att_count += register_recr(net[1], 0, "mid")
#     controller.num_att_layers = cross_att_count


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "CrossAttention":
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def reset_attention_control(model):
    def ca_forward(self):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            temb: Optional[torch.FloatTensor] = None,
        ):
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(  # type: ignore
                    batch_size, self.heads, -1, attention_mask.shape[-1]  # type: ignore
                )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            query = self.to_q(hidden_states)
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            def reshape_heads_to_batch_dim(tensor):
                b, seq_len, dim = tensor.shape
                h = self.heads
                tensor = tensor.reshape(b, seq_len, h, dim // h)
                tensor = tensor.permute(0, 2, 1, 3).reshape(b * h, seq_len, dim // h)
                return tensor

            query = reshape_heads_to_batch_dim(query)
            key = reshape_heads_to_batch_dim(key)
            value = reshape_heads_to_batch_dim(value)

            sim = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
            attn = sim.softmax(dim=-1)
            out = torch.einsum("b i j, b j d -> b i d", attn, value)

            def reshape_batch_dim_to_heads(tensor):
                b, seq_len, dim = tensor.shape
                h = self.heads
                tensor = tensor.reshape(b // h, h, seq_len, dim)
                tensor = tensor.permute(0, 2, 1, 3).reshape(b // h, seq_len, dim * h)
                return tensor

            out = reshape_batch_dim_to_heads(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)
            out = out / self.rescale_output_factor

            return out

        return forward

    def register_recr(net_):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward(net_)
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                register_recr(net__)

    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            register_recr(net[1])
        elif "up" in net[0]:
            register_recr(net[1])
        elif "mid" in net[0]:
            register_recr(net[1])


def init_latent(latent, model, height, width, batch_size):
    latents = latent.expand(
        batch_size, model.unet.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale):
    latents_input = torch.cat([latents] * 2)
    noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


class AttentionRefine(AttentionStore):
    """Simplified AttentionRefine: interpolate edit cross-attn columns with base columns.

    - Maps edit class token columns to base class token columns (1:1 mapping).
    - Uses a linear ramp alpha over diffusion steps (from 0 to 1).
    - Stores attention maps for later aggregation via utils.aggregate_attention.
    """

    def __init__(
        self,
        res: int,
        num_steps: int,
        base_inds: List[int],
        edit_inds: List[int],
        alpha_end: float = 1.0,
    ):
        super().__init__(res)
        self.batch_size = 2
        self.num_steps = max(1, num_steps)
        self.base_inds = base_inds
        self.edit_inds = edit_inds
        self.alpha_end = alpha_end

    def _alpha(self):
        # Linear schedule over attack steps
        s = min(self.cur_step, self.num_steps - 1)
        return (s + 1) / self.num_steps * self.alpha_end

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # record attention for visualization/regularization
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.res // 16) ** 2:
            self.step_store[key].append(attn)

        if not is_cross:
            return attn

        if len(self.base_inds) == 0 or len(self.edit_inds) != len(self.base_inds):
            return attn

        bsz = self.batch_size
        h = attn.shape[0] // bsz
        attn_ = attn.reshape(bsz, h, *attn.shape[1:])
        attn_base, attn_edit = attn_[0], attn_[1]
        alpha = self._alpha()
        base_idx = torch.as_tensor(self.base_inds, device=attn.device)
        edit_idx = torch.as_tensor(self.edit_inds, device=attn.device)
        # Interpolate specific token columns
        blended = (
            alpha * attn_base[:, :, :, base_idx]
            + (1.0 - alpha) * attn_edit[:, :, :, edit_idx]
        )
        attn_edit[:, :, :, edit_idx] = blended
        attn_[1] = attn_edit
        attn = attn_.reshape(bsz * h, *attn_.shape[2:])
        return attn


# ========== Utility for token indices ==========


def find_subsequence(a: List[int], b: List[int]) -> int:
    """Find b as contiguous subsequence inside a. Returns start idx or -1."""
    if len(b) == 0 or len(a) < len(b):
        return -1
    for i in range(len(a) - len(b) + 1):
        if a[i : i + len(b)] == b:
            return i
    return -1


def get_class_and_edit_indices(
    tokenizer, class_text: str, edit_text: str
) -> Tuple[List[int], List[int], List[int]]:
    """
    Returns (orig_class_inds, edit_class_inds, edit_modifier_inds)
    - orig_class_inds: indices [1..L-2] within tokens of the class_text prompt
    - edit_class_inds: matched indices inside edit_text that correspond to the class tokens
    - edit_modifier_inds: indices in edit_text tokens that are not part of class tokens (excluding BOS/EOS)
    """
    tokens_class = tokenizer.encode(class_text)  # includes BOS/EOS
    cls_core = tokens_class[1:-1]
    tokens_edit = tokenizer.encode(edit_text)
    # locate class subsequence inside edit prompt
    start = find_subsequence(tokens_edit, cls_core)
    edit_cls_inds = list(range(start, start + len(cls_core))) if start >= 0 else []
    orig_cls_inds = list(range(1, 1 + len(cls_core)))
    all_edit_inds = set(range(1, len(tokens_edit) - 1))  # drop BOS/EOS
    modifier_inds = sorted(list(all_edit_inds.difference(set(edit_cls_inds))))
    return orig_cls_inds, edit_cls_inds, modifier_inds


# ========== Main attack (P2P losses + attack loss) ==========


@torch.enable_grad()
def diffattack(
    model,
    label,
    controller_unused=None,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    image=None,
    model_name="inception",
    save_path=r"outputs",
    res=224,
    start_step=15,
    iterations=30,
    verbose=True,
    edit_attr: str = "brown",
    args=None,
):
    # dataset labels
    if args.dataset_name == "imagenet_compatible":  # type: ignore
        from dataset_caption import imagenet_label
    elif args.dataset_name == "cub_200_2011":  # type: ignore
        from dataset_caption import CUB_label as imagenet_label
    elif args.dataset_name == "standford_car":  # type: ignore
        from dataset_caption import stanfordCar_label as imagenet_label
    else:
        raise NotImplementedError

    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    # Clean evaluation
    test_image = image.resize((height, height), resample=Image.LANCZOS)  # type: ignore
    test_image = np.float32(test_image) / 255.0
    test_image = test_image[:, :, :3]  # type: ignore
    test_image[:, :] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
    test_image[:, :] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
    test_image = test_image.transpose((2, 0, 1))
    test_image = torch.from_numpy(test_image).unsqueeze(0)

    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(
        label
    )
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    # Build prompts: base class and edited with attribute
    class_text = imagenet_label.refined_Label[label.item()]
    edit_text = f"{edit_attr} {class_text}".strip()
    prompts = [class_text, edit_text]
    print(f"prompts: base='{class_text}' | edit='{edit_text}'")

    # -------- DDIM inversion --------
    latent, inversion_latents = ddim_reverse_sample(
        image, [class_text], model, num_inference_steps, 0, res=height
    )
    inversion_latents = inversion_latents[::-1]

    init_prompt = [class_text, class_text]
    batch_size = len(init_prompt)
    latent = inversion_latents[start_step - 1]

    # Good initial reconstruction by optimizing unconditional embeddings
    max_length = 77
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    text_input = model.tokenizer(
        init_prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)

    uncond_embeddings.requires_grad_(True)
    optimizer_ue = optim.AdamW([uncond_embeddings], lr=1e-1)
    mse = torch.nn.MSELoss()

    context = torch.cat([uncond_embeddings, text_embeddings])

    for ind, t in enumerate(
        tqdm(
            model.scheduler.timesteps[1 + start_step - 1 :],
            desc="Optimize_uncond_embed",
        )
    ):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer_ue.zero_grad()
            loss = mse(out_latents, inversion_latents[start_step - 1 + ind + 1])
            loss.backward()
            optimizer_ue.step()
            context = torch.cat([uncond_embeddings, text_embeddings])
        with torch.no_grad():
            latents = diffusion_step(
                model, latents, context, t, guidance_scale
            ).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    # -------- P2P controller & contexts (attack) --------
    # Token indices mapping for P2P keep (class tokens)
    base_cls_inds, edit_cls_inds, edit_modifier_inds = get_class_and_edit_indices(
        model.tokenizer, class_text, edit_text
    )
    controller = AttentionRefine(
        res=args.res,
        num_steps=len(model.scheduler.timesteps[1 + start_step - 1 :]),
        base_inds=base_cls_inds,
        edit_inds=edit_cls_inds,
        alpha_end=1.0,
    )
    register_attention_control(model, controller)

    # contexts per step
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    context = [
        [torch.cat([all_uncond_emb[i]] * 2), text_embeddings]
        for i in range(len(all_uncond_emb))
    ]
    context = [torch.cat(i) for i in context]

    original_latent = latent.clone()
    latent.requires_grad_(True)
    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    # Pseudo mask
    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    # Loss weights (reuse args weights for compatibility)
    w_attack = args.attack_loss_weight
    w_p2p_keep = max(
        1, args.cross_attn_loss_weight // 10
    )  # tighter than original variance term
    w_p2p_align = max(1, args.cross_attn_loss_weight // 20)

    pbar = tqdm(range(iterations), desc="Iterations-P2P")
    for _, _ in enumerate(pbar):
        controller.reset()

        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1 :]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

        # Aggregate cross-attn maps for base (select=0) and edit (select=1)
        before_cross = aggregate_attention(
            prompts, controller, args.res // 32, ("up", "down"), True, 0, is_cpu=False
        )
        after_cross = aggregate_attention(
            prompts, controller, args.res // 32, ("up", "down"), True, 1, is_cpu=False
        )

        # Build pseudo mask once if enabled
        if init_mask is None:
            if len(base_cls_inds) > 0:
                base_idx = torch.as_tensor(base_cls_inds, device=before_cross.device)
                region = before_cross[:, :, base_idx].mean(-1)
            else:
                region = before_cross.mean(-1)
            region = region / (region.max() + 1e-12)
            init_mask = F.interpolate(
                region.unsqueeze(0).unsqueeze(0), init_image.shape[-2:], mode="bilinear"
            ).clamp(0, 1)
            if hard_mask:
                init_mask = init_mask.gt(0.5).float()

        # Build P2P losses
        device = before_cross.device
        if len(base_cls_inds) > 0 and len(edit_cls_inds) == len(base_cls_inds):
            base_idx = torch.as_tensor(base_cls_inds, device=device)
            edit_idx = torch.as_tensor(edit_cls_inds, device=device)
            # keep common tokens similar
            L_keep = (
                (after_cross[:, :, edit_idx] - before_cross[:, :, base_idx])
                .abs()
                .mean()
            )
        else:
            L_keep = torch.zeros((), device=device)

        if len(edit_modifier_inds) > 0:
            edit_mod_idx = torch.as_tensor(edit_modifier_inds, device=device)
            # align modifier tokens to base class region (encourage overlap)
            base_region = (
                before_cross[:, :, base_idx].mean(-1, keepdim=True)
                if len(base_cls_inds) > 0
                else before_cross.mean(-1, keepdim=True)
            )
            mod_maps = after_cross[:, :, edit_mod_idx]
            overlap = (mod_maps * base_region).mean()
            L_align = -overlap  # maximize overlap
        else:
            L_align = torch.zeros((), device=device)

        # Decode current image for classifier
        init_out_image = (
            model.vae.decode(1 / 0.18215 * latents)["sample"][1:]
            * (init_mask if init_mask is not None else 1)
            + (1 - (init_mask if init_mask is not None else 1)) * init_image
        )
        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor(
            [0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device
        )
        std = torch.as_tensor(
            [0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device
        )
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        if args.dataset_name != "imagenet_compatible":  # type: ignore
            pred = classifier(out_image) / 10
        else:
            pred = classifier(out_image)

        attack_loss = -cross_entro(pred, label)

        loss = w_attack * attack_loss + w_p2p_keep * L_keep + w_p2p_align * L_align

        if verbose:
            pbar.set_postfix_str(
                f"atk: {attack_loss.item():.4f} keep: {L_keep.item():.4f} align: {L_align.item():.4f} total: {loss.item():.4f}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # -------- Final decode & save --------
    with torch.no_grad():
        controller.reset()
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1 :]):
            latents = diffusion_step(model, latents, context[ind], t, guidance_scale)

    out_image = (
        model.vae.decode(1 / 0.18215 * latents.detach())["sample"][1:]
        * (init_mask if init_mask is not None else 1)
        + (1 - (init_mask if init_mask is not None else 1)) * init_image
    )
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor(
        [0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device
    )
    std = torch.as_tensor(
        [0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device
    )
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    image_np = latent2image(model.vae, latents.detach())

    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = (
        image_np[1:].astype(np.float32)
        / 255
        * (
            init_mask.squeeze().unsqueeze(-1).cpu().numpy()
            if init_mask is not None
            else 1
        )
        + (
            1
            - (
                init_mask.squeeze().unsqueeze(-1).cpu().numpy()
                if init_mask is not None
                else 0
            )
        )
        * real
    )
    image_np = (perturbed * 255).astype(np.uint8)
    view_images(
        np.concatenate([real, perturbed]) * 255,
        show=False,
        save_path=save_path
        + f"_p2p_diff_{model_name}_image_{'ATKSuccess' if pred_accuracy == 0 else 'Fail'}.png",
    )
    view_images(perturbed * 255, show=False, save_path=save_path + "_p2p_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))
    print(
        "L1: {}\tL2: {}\tLinf: {}".format(
            L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)
        )
    )

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
    view_images(
        diff.clip(0, 255), show=False, save_path=save_path + "_p2p_diff_relative.png"
    )

    diff_abs = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(
        diff_abs.clip(0, 255),
        show=False,
        save_path=save_path + "_p2p_diff_absolute.png",
    )

    reset_attention_control(model)

    return image_np[0], pred_accuracy_clean, pred_accuracy


@torch.enable_grad()
def diffattack_ptp(
    model,
    label,
    prompts,
    ptp_controller,  # 这里传入prompt-to-prompt的AttentionControl实例
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    image=None,
    model_name="inception",
    save_path="output",
    res=224,
    start_step=15,
    iterations=30,
    verbose=True,
    topN=1,
    args=None,
):
    """
    diffattack with prompt-to-prompt style attention controller.
    """
    # 1. 数据和模型准备
    if args.dataset_name == "imagenet_compatible":  # type: ignore
        from dataset_caption import imagenet_label
    elif args.dataset_name == "cub_200_2011":  # type: ignore
        from dataset_caption import CUB_label as imagenet_label
    elif args.dataset_name == "standford_car":  # type: ignore
        from dataset_caption import stanfordCar_label as imagenet_label
    else:
        raise NotImplementedError

    label = torch.from_numpy(label).long().cuda()

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)

    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)

    height = width = res

    # 2. 预处理与DDIM反推
    latent, inversion_latents = ddim_reverse_sample(
        image, prompts, model, num_inference_steps, 0, res=height
    )
    inversion_latents = inversion_latents[::-1]
    latent = inversion_latents[start_step - 1]

    # 3. 优化uncond embedding（可选，流程同原版）
    max_length = 77
    batch_size = len(prompts)
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    all_uncond_emb = []
    latent, latents = init_latent(latent, model, height, width, batch_size)
    uncond_embeddings.requires_grad_(True)
    optimizer = optim.AdamW([uncond_embeddings], lr=1e-1)
    loss_func = torch.nn.MSELoss()
    context = torch.cat([uncond_embeddings, text_embeddings])

    for ind, t in enumerate(
        tqdm(
            model.scheduler.timesteps[1 + start_step - 1 :],
            desc="Optimize_uncond_embed",
        )
    ):
        for _ in range(10 + 2 * ind):
            out_latents = diffusion_step(model, latents, context, t, guidance_scale)
            optimizer.zero_grad()
            loss = loss_func(out_latents, inversion_latents[start_step - 1 + ind + 1])
            loss.backward()
            optimizer.step()
            context = torch.cat([uncond_embeddings, text_embeddings])
        with torch.no_grad():
            latents = diffusion_step(
                model, latents, context, t, guidance_scale
            ).detach()
            all_uncond_emb.append(uncond_embeddings.detach().clone())

    uncond_embeddings.requires_grad_(False)

    # 4. 注册prompt-to-prompt控制器
    register_attention_control(model, ptp_controller)

    # 5. 攻击优化（对latent做梯度攻击，attention由ptp_controller控制）
    original_latent = latent.clone()
    latent.requires_grad_(True)
    optimizer = optim.AdamW([latent], lr=1e-2)
    cross_entro = torch.nn.CrossEntropyLoss()
    init_image = preprocess(image, res)

    apply_mask = args.is_apply_mask
    hard_mask = args.is_hard_mask
    if apply_mask:
        init_mask = None
    else:
        init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()

    pbar = tqdm(range(iterations), desc="Iterations")
    for _, _ in enumerate(pbar):
        ptp_controller.loss = 0
        ptp_controller.reset()
        latents = torch.cat([original_latent, latent])

        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1 :]):
            latents = diffusion_step(model, latents, context, t, guidance_scale)

        # 可选：聚合attention map用于mask
        # 可选：可用ptp_controller的attention store做可视化

        if init_mask is None:
            # 这里可用ptp_controller的attention map做mask
            init_mask = torch.ones([1, 1, *init_image.shape[-2:]]).cuda()
        init_out_image = (
            model.vae.decode(1 / 0.18215 * latents)["sample"][1:] * init_mask
            + (1 - init_mask) * init_image
        )

        out_image = (init_out_image / 2 + 0.5).clamp(0, 1)
        out_image = out_image.permute(0, 2, 3, 1)
        mean = torch.as_tensor(
            [0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device
        )
        std = torch.as_tensor(
            [0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device
        )
        out_image = out_image[:, :, :].sub(mean).div(std)
        out_image = out_image.permute(0, 3, 1, 2)

        if args.dataset_name != "imagenet_compatible":
            pred = classifier(out_image) / 10
        else:
            pred = classifier(out_image)

        attack_loss = -cross_entro(pred, label) * args.attack_loss_weight
        # ptp_controller.loss 可作为结构损失（如有定义）
        self_attn_loss = getattr(ptp_controller, "loss", 0) * args.self_attn_loss_weight
        loss = attack_loss + self_attn_loss

        if verbose:
            pbar.set_postfix_str(
                f"attack_loss: {attack_loss.item():.5f} "
                f"self_attn_loss: {self_attn_loss.item() if isinstance(self_attn_loss, torch.Tensor) else self_attn_loss:.5f} "
                f"loss: {loss.item():.5f}"
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 6. 最终生成与评估
    with torch.no_grad():
        ptp_controller.loss = 0
        ptp_controller.reset()
        latents = torch.cat([original_latent, latent])
        for ind, t in enumerate(model.scheduler.timesteps[1 + start_step - 1 :]):
            latents = diffusion_step(model, latents, context, t, guidance_scale)

    out_image = (
        model.vae.decode(1 / 0.18215 * latents.detach())["sample"][1:] * init_mask
        + (1 - init_mask) * init_image
    )
    out_image = (out_image / 2 + 0.5).clamp(0, 1)
    out_image = out_image.permute(0, 2, 3, 1)
    mean = torch.as_tensor(
        [0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device
    )
    std = torch.as_tensor(
        [0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device
    )
    out_image = out_image[:, :, :].sub(mean).div(std)
    out_image = out_image.permute(0, 3, 1, 2)

    pred = classifier(out_image)
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))

    logit = torch.nn.Softmax()(pred)
    print("after_pred:", pred_label, logit[0, pred_label[0]])
    print("after_true:", label, logit[0, label[0]])

    # 可选：可视化与距离度量
    image = latent2image(model.vae, latents.detach())
    real = (init_image / 2 + 0.5).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    perturbed = (
        image[1:].astype(np.float32)
        / 255
        * init_mask.squeeze().unsqueeze(-1).cpu().numpy()
        + (1 - init_mask.squeeze().unsqueeze(-1).cpu().numpy()) * real
    )
    image = (perturbed * 255).astype(np.uint8)
    view_images(
        np.concatenate([real, perturbed]) * 255,
        show=False,
        save_path=save_path
        + "_diff_{}_image_{}.png".format(
            model_name, "ATKSuccess" if pred_accuracy == 0 else "Fail"
        ),
    )
    view_images(perturbed * 255, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))
    print(
        "L1: {}\tL2: {}\tLinf: {}".format(
            L1(real, perturbed), L2(real, perturbed), Linf(real, perturbed)
        )
    )

    diff = perturbed - real
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255
    view_images(
        diff.clip(0, 255), show=False, save_path=save_path + "_diff_relative.png"
    )
    diff = (np.abs(perturbed - real) * 255).astype(np.uint8)
    view_images(
        diff.clip(0, 255), show=False, save_path=save_path + "_diff_absolute.png"
    )

    reset_attention_control(model)
    return image[0], 0, 0
