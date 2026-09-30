"""Rex-base 的推理网络：DeBERTa 编码器 + 实体位置连线预测头。"""
import torch
from torch import nn

from .rotary import RotaryEmbedding
from .utils_mod import DebertaV2Model2dAttnMask


class RexModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.model_type != 'deberta-v2':
            raise ValueError('本模块只支持随包提供的 DeBERTa-v2 Rex-base 权重')
        # 这里只创建网络，全部参数随后由 model_loader 一次性从本地权重加载。
        self.plm = DebertaV2Model2dAttnMask(config)
        self.ffnq = nn.Linear(config.hidden_size, config.hidden_size)
        self.ffnk = nn.Linear(config.hidden_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(config.hidden_size)

    def forward(self, input_ids, attention_masks, token_type_ids, position_ids):
        sequence = self.plm(input_ids, attention_mask=attention_masks,
                            token_type_ids=token_type_ids, position_ids=position_ids)[0]
        q = self.rotary_emb.rotate_queries_or_keys(self.ffnq(sequence), position_ids=position_ids)
        k = self.rotary_emb.rotate_queries_or_keys(self.ffnk(sequence), position_ids=position_ids)
        logits = torch.bmm(q, k.permute(0, 2, 1))
        # 沿用原实现的阈值：logit > 0，等价于 sigmoid 概率 > 0.5。
        return torch.where(logits > 0)
