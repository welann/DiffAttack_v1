# 基于 Prompt-to-Prompt 的损失改进思路

本文围绕现有 `loss = self_attn_loss + attack_loss + variance_cross_attn_loss` 提供可实施的优化方向，结合 `prompt-to-prompt_stable.ipynb` 的注意力编辑思想，增强可迁移性与稳定性。

## 现状与问题
- attack_loss：当前为交叉熵（或其负号），对极端防御/分布偏移鲁棒性有限，梯度在高置信时不稳定。
- variance_cross_attn_loss：最小化交叉注意力方差仅鼓励“均匀化”，未显式把注意力从“真类”转向“干扰类”，且忽略层/头差异。
- self_attn_loss：MSE 全层同权，未区分不同尺度结构的重要性；对纹理与边缘的保持不够细粒度。

## 交叉注意力项（结合 Prompt-to-Prompt）
1) 注意力替换损失 L_replace（P2P 核心）
- 思想：在指定层与时间窗，将“后”注意力与“干扰提示”对应注意力对齐，弱化真类语义。
- 做法：构造干扰提示 `prompt_tgt`（近邻类别/上下文干扰），前向得到 `A_tgt`，对齐同层同 token：
  L_replace = || A_after_true(tokens_true) - A_tgt(tokens_tgt) ||_1/2

2) 对比注意力损失 L_contrast（指向性转移）
- 将注意力质量从真类 token 推向干扰 token：
  L_contrast = ReLU(margin - mean(A_after[tgt]) + mean(A_after[true]))
- 可结合 head/layer 加权，仅在 P2P 推荐的中高层（mid/up）生效。

3) 熵正则替代方差 L_entropy
- 目标：提升交叉注意力分布熵，避免方差对异常值敏感：
  L_entropy = - H(softmax(A_after_true))
- 可叠加 head-dropout 与时序平滑项，稳定优化。

4) 空间掩模加权（更精准）
- 利用“前”真类注意力/Grad-CAM 得到对象掩模 M，仅在 M 内强化 L_contrast/L_replace；M 外弱化，使扰动集中于关键语义。

## 自注意力项（结构保持）
1) 分层权重与度量替换
- 上采样层与高分辨层权重大（纹理细节），中层次之下权重小；
- 将 MSE 替换为 KL/JS（分布更稳）：L_self = Σ_w_l D(attn_after^l || attn_before^l)。

2) 频域结构正则（可选）
- 在掩模 M 内约束拉普拉斯/小波能量偏移，抑制高频伪影：L_freq = ||Lap(img_after) - Lap(img_before)||_1。

## 攻击项（更强可迁移）
1) 采用 Margin/CW 型目标
- L_attack = max(0, logit_true - max_{k≠true} logit_k + κ) 或 CW 损失，梯度信号更稳定。

2) EOT 与模型集成
- 对输入/尺度/色彩做随机变换的期望梯度（EOT）；
- 多模型加权或最坏情况目标：L_attack = Σ_j w_j L_j 或 max_j L_j。

3) 分布式目标
- 用 JS/CE 逼近“目标分布”（如均匀或干扰类别 peaked 分布），弱化过拟合单一决策边界。

## 联合目标与权重调度
- 迭代/时间步调度：早期强调 self（结构保真），中后期提升 contrast/replace（语义转移）与 attack；
- 示例：w_self(t) 递减，w_contrast(t)、w_replace(t)、w_attack(t) 递增；仅在 mid/up 层、t ∈ [t1, t2] 启用替换/对比项。

## 伪代码片段（与现有变量对齐）
```python
# 已有: before_true_label_attention_map, after_true_label_attention_map
# 额外: A_tgt 来自 prompt_tgt 的一次前向（P2P），对齐相同层/位置

# 熵正则（替代方差）
L_entropy = -(after_true_label_attention_map.softmax(-1) *
              after_true_label_attention_map.log_softmax(-1)).sum(-1).mean()

# 对比注意力（真类 vs 干扰）
A_true = after_true_label_attention_map.mean([-3,-2])  # 空间平均
A_tgt  = A_tgt_selected.mean([-3,-2])                  # 干扰 token 对应
L_contrast = torch.relu(margin - A_tgt.mean() + A_true.mean())

# 注意力替换（P2P）
L_replace = (after_true_label_attention_map - A_tgt_selected).abs().mean()

# 新总损失（含调度）
loss = w_self*L_self + w_attack*L_attack + w_ent*L_entropy + \
       w_contrast*L_contrast + w_replace*L_replace
```

## 实施优先级（建议顺序）
1) 用熵正则替代方差，并加入对比注意力（低成本显著增益）。
2) 将 CE 改为 Margin/CW，并加入 EOT/小型模型集成。
3) 引入 P2P 的注意力替换，与干扰提示结合的 L_replace。
4) 自注意力改为分层 KL/JS，并在高分辨层加大权重；必要时加入频域正则。

## 自适应集成分类损失（AdaEA）
为提升黑盒迁移与梯度稳定性，建议用自适应集成（AdaEA）替换单模型交叉熵：

1) 目标形式（以无目标为例）
- Margin/CW 型更稳健：`L_i = logit_true − max_{k≠true} logit_k + κ`（取 ReLU 或直接最小化 margin）；目标攻击则最大化目标类 logit margin。

2) 动态权重与梯度融合
- 梯度归一：每模型梯度 `g_i ← g_i / (||g_i||_2 + ε)`，避免单模型主导；
- 相似度加权：先取临时合梯度 `ḡ = Σ g_i / N`，计算 `s_i = cos(g_i, ḡ)`，`w_i ∝ softmax(τ·s_i)`（τ≈5–10）；
- 或按最近若干步的 margin 改善量 Δmargin_i 做 softmax 加权，配合 EMA 稳定；
- 动量：`m ← μ·m + (Σ w_i·g_i)`，更新用 `m`（μ≈0.9）；
- 总体裁剪：对 `m` 做 `clip_norm`，防止数值不稳。

3) EOT 稳健性
- 对解码图像施加轻量随机变换（旋转/缩放/色偏），取期望梯度，隔步启用以控算力。

4) 伪代码（与现有变量对齐）
```python
# 替换单模型：diff_latent_attack.py:381 的 classifier 定义改为 ensemble 列表
# 在采样迭代（约 560 行后）替换 attack_loss 计算：

logits_list, grads = [], []
for clf in ensemble:
    logits = clf(out_image)  # [B, C]
    logits_list.append(logits)
    loss_i = margin_loss(logits, label)  # 无目标/目标皆可
    g_i = torch.autograd.grad(loss_i, latent, retain_graph=True)[0]
    g_i = g_i / (g_i.norm(p=2) + 1e-12)
    grads.append(g_i)

# 相似度加权
g_bar = torch.stack(grads, 0).mean(0).detach()
s = [torch.nn.functional.cosine_similarity(g, g_bar, dim=None).mean() for g in grads]
w = torch.softmax(torch.tensor(s) * tau, dim=0).to(latent.device)

# 动量融合
state.setdefault('m', torch.zeros_like(latent))
state['m'] = mu * state['m'] + sum(w_i * g_i for w_i, g_i in zip(w, grads))

L_cls_ens = sum(w_i * margin_loss(lg, label) for w_i, lg in zip(w, logits_list))
```

## 联合目标与权重调度（更新）
- 总损失：`L_total = λ_cls·L_cls_ens + λ_ent·L_entropy + λ_contrast·L_contrast + λ_replace·L_replace + λ_self·L_self [+ λ_freq·L_freq]`；
- 建议系数：`λ_cls=1.0, λ_ent=0.1–0.3, λ_contrast=0.2–0.5, λ_replace=0.2–0.5, λ_self=0.3–0.7`；如需更强不可察觉性，可再加 `λ_lpips·LPIPS + λ_l2·L2`；
- 迭代/时间步调度：早期强调 `L_self`（结构保真），中后期提升 `L_contrast/L_replace/L_cls`；仅在 mid/up 层、t ∈ [t1, t2] 启用替换/对比项。

## 实施位置与文件指引
- `diff_latent_attack.py:381`：单模型 `classifier` 的定义位置，改为组装 `ensemble` 列表；
- `diff_latent_attack.py:560` 起：已有的注意力汇聚结果与 `attack_loss`/`variance_cross_attn_loss`/`self_attn_loss` 计算；
  - 用 `L_entropy/L_contrast/L_replace` 替换 `variance_cross_attn_loss`；
  - 用 `L_cls_ens`（AdaEA）替换单一 `-CE` 的 `attack_loss`；
- `diff_latent_attack-0.9.0.py:485–501`：同样的损失位置，含 “todo 把这个loss改了” 注释，可优先在此版本先落地。
