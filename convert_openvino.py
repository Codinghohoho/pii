"""Convert the local Rex checkpoint to an OpenVINO FP32 inference model.

Run from the pii directory with PYTHONPATH=.. .venv/bin/python -m pii.convert_openvino.
"""
from pathlib import Path

import openvino as ov
import torch

from .config import MODEL_PATH, REX_SCHEMA
from .model_loader import RexBaseRuntime
from .rex_network.position import build_position_ids_attn_mask


class DenseRex(torch.nn.Module):
    """Expose fixed-shape logits; entity coordinates remain a post-processing step."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_masks, token_type_ids, position_ids):
        sequence = self.model.plm(
            input_ids, attention_mask=attention_masks,
            token_type_ids=token_type_ids, position_ids=position_ids,
        )[0]
        q = self.model.rotary_emb.rotate_queries_or_keys(
            self.model.ffnq(sequence), position_ids=position_ids,
        )
        k = self.model.rotary_emb.rotate_queries_or_keys(
            self.model.ffnk(sequence), position_ids=position_ids,
        )
        return torch.bmm(q, k.permute(0, 2, 1))


def main():
    # 转换阶段必须从原始 PyTorch 权重读取，不能使用默认的 OpenVINO 后端。
    runtime = RexBaseRuntime(MODEL_PATH, backend='pytorch')
    tokenizer = runtime.rex_tokenizer
    hint = runtime.inference.encode_schema(REX_SCHEMA)[1][0]
    encoded = tokenizer(hint, '患者姓名：张三', return_token_type_ids=True)
    ids = encoded['input_ids']
    types = encoded['token_type_ids']
    positions, attention = build_position_ids_attn_mask(
        tokenizer, ids, types, encoded['attention_mask'],
    )
    inputs = tuple(torch.tensor([value], dtype=torch.long) for value in (
        ids, attention, types, positions,
    ))
    with torch.inference_mode():
        model = ov.convert_model(DenseRex(runtime.model).eval(), example_input=inputs)
    model.reshape({port: ov.PartialShape([1, -1, -1] if index == 1 else [1, -1])
                   for index, port in enumerate(model.inputs)})
    output = Path(__file__).resolve().parent / 'model' / 'rex-openvino' / 'rex.xml'
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, output, compress_to_fp16=False)
    print(output)


if __name__ == '__main__':
    main()
