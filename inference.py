"""从原 Rex 实现拆出的实体抽取流程；仅支持当前使用的平铺 schema。

流程：拼 schema → 分词和构造注意力 → 网络预测 → 换算字符位置。
不包含训练、评估、分类任务或多层关系抽取。
"""
from collections import defaultdict

import torch

from .rex_network.position import build_position_ids_attn_mask
from .rex_network.token_config import PREFIX_TOKEN, TYPE_TOKEN


class RexInference:
    def __init__(self, tokenizer, model, device):
        """创建一次 Rex 推理器。

        输入：``tokenizer`` 分词器、``model`` Rex 神经网络、``device``（CPU/CUDA）。
        输出：无；三个对象保存到实例中供后续 ``extract`` 使用。
        """
        self.tokenizer, self.model, self.device = tokenizer, model, device

    def encode_schema(self, labels):
        """生成 schema 提示词及字段标记位置。

        输入：``labels``，字段名称列表，例如 ``['患者姓名', '医生姓名']``。
        输出：``(maps, hints)``；``hints`` 是给模型的提示词列表，``maps``
        是同下标提示词的“字段名 -> TYPE_TOKEN 位置”字典。每组提示词最多约
        256 token，过多字段会自动分组。
        """
        hints, maps = [], []
        hint, positions, size = '', {}, 2
        prefix_size = len(self.tokenizer('')['input_ids']) - 1
        for label in sorted(set(labels)):
            label_size = len(self.tokenizer(label)['input_ids']) - 1
            if label_size > 125:
                raise ValueError(f'schema 字段过长：{label}')
            extra = label_size + (prefix_size if not hint else 0)
            if hint and size + extra > 256:
                hints.append(hint)
                maps.append(positions)
                hint, positions, size = '', {}, 2
            if not hint:
                hint = PREFIX_TOKEN
                size += prefix_size
            positions[label] = len(hint)
            hint += TYPE_TOKEN + label
            size += label_size
        if hint:
            hints.append(hint)
            maps.append(positions)
        return maps, hints

    def extract(self, text, labels):
        """在一个正文窗口中运行模型并提取实体。

        输入：``text``，一个窗口的原文；``labels``，要抽取的字段名列表。
        输出：实体列表，每项为 ``{'label': 字段名, 'text': 原文片段,
        'offset': [start, end]}``。坐标是当前窗口内的 Python 字符坐标，
        不是全文坐标；全文分窗和坐标相加由 ``rex.py`` 负责。
        """
        maps, hints = self.encode_schema(labels)
        entities = []
        for hint, field_positions in zip(hints, maps):
            encoded = self.tokenizer(
                hint, text, truncation='only_second', max_length=512, stride=32,
                return_overflowing_tokens=True, return_token_type_ids=True,
                return_offsets_mapping=True,
            )
            for ids, types, mask, offsets in zip(
                encoded['input_ids'], encoded['token_type_ids'],
                encoded['attention_mask'], encoded['offset_mapping'],
            ):
                if not any(types):
                    # 部分分词器不提供句段标记，按 schema 后的分隔符重建。
                    types, previous, current = [], -1, 0
                    for token in ids:
                        if previous == self.tokenizer.sep_token_id and token != self.tokenizer.sep_token_id:
                            current = 1
                        types.append(current)
                        previous = token
                positions, attention = build_position_ids_attn_mask(self.tokenizer, ids, types, mask)
                inputs = [torch.tensor([value], dtype=torch.long, device=self.device)
                          for value in (ids, attention, types, positions)]
                _, rows, cols = self.model(*inputs)
                entities.extend(self.decode(text, types, offsets, field_positions,
                                            rows.tolist(), cols.tolist()))
        return entities

    @staticmethod
    def decode(text, types, offsets, field_positions, rows, cols):
        """把模型预测的 token 连线还原成实体。

        输入：``text`` 原文；``types`` token 属于 schema/正文的标记；
        ``offsets`` 每个 token 对应的字符范围；``field_positions`` 字段标记位置；
        ``rows``、``cols`` 模型预测的 token 连线端点。
        输出：实体字典列表，格式为 ``label``、``text``、``offset``。

        一个实体必须同时满足：正文首 token 连到正文尾 token、首 token 连到
        字段标记、字段标记连到尾 token。这样可以判断“找到的文字属于哪个字段”。
        """
        hint_count = types.count(0)
        char_to_token = {offsets[i][0]: i for i in range(hint_count)}
        fields = {label: char_to_token[pos] for label, pos in field_positions.items()}
        token_to_field = {token: label for label, token in fields.items()}
        heads, tails, spans = defaultdict(set), defaultdict(set), []
        for head, tail in zip(rows, cols):
            if head >= hint_count and tail >= hint_count and head <= tail:
                spans.append((head, tail))
            if head >= hint_count and tail in token_to_field:
                heads[token_to_field[tail]].add(head)
            if tail >= hint_count and head in token_to_field:
                tails[token_to_field[head]].add(tail)
        entities = []
        for head, tail in spans:
            for label in fields:
                if head in heads[label] and tail in tails[label]:
                    start, end = offsets[head][0], offsets[tail][1]
                    # 与原实现一致：去掉实体开头的普通空格，保留原文坐标。
                    while start < len(text) and text[start] == ' ':
                        start += 1
                    if 0 <= start < end <= len(text):
                        entities.append(dict(label=label, text=text[start:end], offset=[start, end]))
        return entities
