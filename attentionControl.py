from typing import Union, Tuple
import torch
import abc


class AttentionControl(abc.ABC):
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def between_steps(self):
        return

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= 0:
            h = attn.shape[0]
            self.forward(attn[h // 2 :], is_cross, place_in_unet)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn


class AttentionStore(AttentionControl):
    def __init__(self, res):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.res = res

    @staticmethod
    def get_empty_store():
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= (self.res // 16) ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] = (
                        self.step_store[key][i] + self.attention_store[key][i]
                    )
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttentionControlEdit(AttentionStore, abc.ABC):
    def __init__(
        self, num_steps: int, self_replace_steps: Union[float, Tuple[float, float]], res
    ):
        super(AttentionControlEdit, self).__init__(res)
        self.batch_size = 2
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = (
            # self_replace_steps  是控制自注意力替换的时间范围
            int(num_steps * self_replace_steps[0]),
            int(num_steps * self_replace_steps[1]),
        )
        # 用于日志的累计（float）
        self.loss = 0.0
        self.cross_loss = 0.0
        # 用于本次优化步的 tensor 累计（会在外部取出并清空，避免跨步保留计算图）
        self._loss_tensor = None
        self._cross_loss_tensor = None
        self.criterion = torch.nn.MSELoss()

    # todo 需要在这里计算好注意力图不同得到的loss，看utils文件里的计算，这里的注意力图会存储原始的prompt中的注意力权重
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        # 2. 判断是否需要处理（交叉注意力或在替换步骤范围内的自注意力）
        if is_cross or (
            self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]
        ):
            # 3. 重塑张量以分离基础注意力和替换注意力
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            # if is_cross:
                # """
                # ==========================================
                # ======= Cross Attention Control ==========
                # === 将指定词的注意力进行处理, 通过loss拉近原图与新图之间的距离 ==
                # ==========================================
                # """
                # # print("Cross Attention Control at step:", self.cur_step)
                # # print(
                # #     f"Token index from {self.first_token_index} to {self.last_token_index}"
                # # )
                # # print(
                # #     f"Attention shape: {attn.shape}, base: {attn_base.shape}, replace: {attn_repalce.shape}"
                # # )

                # token_index = self.first_token_index  # 需要替换的token索引
                # token_index_end = self.last_token_index  # 需要替换的token索引结束位置
                # token_attention_base = attn_base[
                #     :, :, token_index:token_index_end
                # ]  # 基础注意力中对应token的注意力
                # token_attention_replace = attn_repalce[
                #     :, :, :, token_index:token_index_end
                # ]  # 替换注意力中对应token的注意力

                # # 计算当前 forward 的 tensor loss（用于梯度），累加到临时 tensor
                # _l = self.criterion(
                #     token_attention_base.unsqueeze(0), token_attention_replace
                # )
                # if self._cross_loss_tensor is None:
                #     self._cross_loss_tensor = _l
                # else:
                #     self._cross_loss_tensor = self._cross_loss_tensor + _l

                # self.cross_loss += _l.detach().item()

                # print(f"Step {self.cur_step}: Cross Attention Loss: {self.cross_loss}")

            # 4. 对自注意力进行特殊处理
            if not is_cross:
                """
                ==========================================
                ========= Self Attention Control =========
                === Details please refer to Section 3.4 ==
                ==========================================
                """
                _l2 = self.criterion(
                    attn[1:], self.replace_self_attention(attn_base, attn_repalce)
                )
                if self._loss_tensor is None:
                    self._loss_tensor = _l2
                else:
                    self._loss_tensor = self._loss_tensor + _l2
                self.loss += _l2.detach().item()
                # print(f"Step {self.cur_step}: Self Attention Loss: {self.loss}")

            self.pop_current_attn_loss()
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])

        return attn

    def replace_self_attention(self, attn_base, att_replace):
        return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)

    def pop_current_attn_loss(self):
            """
            返回当前迭代（或一次生成流程）累积的 attention loss tensor（可直接加到总 loss 并 backward），
            并清空内部 tensor（避免下一次 backward 时重复使用已释放的中间节点）。
            返回 None 或 tensor。
            """
            # 清空临时 tensor（注意：日志累积 self.cross_loss / self.loss 保留）
            self._cross_loss_tensor = None
            self._loss_tensor = None