"""加载 Rex 模型并提供统一的窗口推理接口。"""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from .config import MODEL_PATH, OPENVINO_MODEL_PATH, REX_BACKEND, REX_DEVICE, REX_THREADS
from .inference import RexInference
from .rex_network.model import RexModel
from .rex_network.position import build_position_ids_attn_mask
from .rex_network.token_config import (
    PREFIX_TOKEN, TYPE_TOKEN, CLASSIFY_TOKEN, MULTI_CLASSIFY_TOKEN,
)


class OpenVinoRexInference(RexInference):
    """用 OpenVINO 执行密集连线预测，再复用 Rex 的实体解码逻辑。"""

    def __init__(self, tokenizer, compiled_model):
        super().__init__(tokenizer, model=None, device='CPU')
        self.compiled_model = compiled_model

    def extract(self, text, labels):
        """返回与 ``RexInference.extract`` 相同格式的实体列表。"""
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
                    # 与 PyTorch 路径保持一致，兼容不提供句段标记的分词器。
                    types, previous, current = [], -1, 0
                    for token in ids:
                        if previous == self.tokenizer.sep_token_id and token != self.tokenizer.sep_token_id:
                            current = 1
                        types.append(current)
                        previous = token
                positions, attention = build_position_ids_attn_mask(
                    self.tokenizer, ids, types, mask,
                )
                inputs = [
                    np.asarray([value], dtype=np.int64)
                    for value in (ids, attention, types, positions)
                ]
                output = self.compiled_model(inputs)
                logits = output[self.compiled_model.outputs[0]][0]
                rows, cols = np.where(logits > 0)
                entities.extend(self.decode(
                    text, types, offsets, field_positions,
                    rows.tolist(), cols.tolist(),
                ))
        return entities


class RexBaseRuntime:
    """模型和分词器只加载一次，后续文本窗口复用同一对象。"""

    def __init__(self, model_path=MODEL_PATH, backend=REX_BACKEND,
                 device=REX_DEVICE, threads=REX_THREADS):
        """加载 Rex 模型。

        默认使用 ``model/rex-openvino/rex.xml`` 的 OpenVINO GPU 模型。
        CPU 后端使用 2 个线程；GPU 后端不使用 CPU 线程参数。
        设置 ``backend='pytorch'`` 可使用原始 PyTorch 权重进行对照或回退。
        """
        model_path = Path(model_path)
        backend = backend.lower()
        if backend not in {'openvino', 'pytorch'}:
            raise ValueError(f'不支持的 Rex 后端：{backend}')
        self.backend = backend
        self.device_name = str(device).upper()
        self.threads = int(threads)
        self.rex_tokenizer = self._load_tokenizer(model_path)
        if backend == 'openvino':
            self._load_openvino(self.device_name)
        else:
            self._load_pytorch(model_path)

    @staticmethod
    def _load_tokenizer(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # 四个特殊 token 的顺序必须与原始权重一致。
        tokenizer.add_special_tokens({'additional_special_tokens': [
            PREFIX_TOKEN, TYPE_TOKEN, CLASSIFY_TOKEN, MULTI_CLASSIFY_TOKEN,
        ]})
        return tokenizer

    def _load_openvino(self, device):
        import openvino as ov

        ov_path = OPENVINO_MODEL_PATH
        if not ov_path.is_file():
            raise FileNotFoundError(
                f'OpenVINO Rex 模型不存在：{ov_path}；请先运行 '
                'PYTHONPATH=.. .venv/bin/python -m pii.convert_openvino，'
                '或设置 REX_BACKEND=pytorch 使用原始权重'
            )
        core = ov.Core()
        model = core.read_model(ov_path)
        properties = {'PERFORMANCE_HINT': 'LATENCY'}
        if device == 'CPU':
            properties['INFERENCE_NUM_THREADS'] = self.threads
        try:
            self.compiled_model = core.compile_model(model, device, properties)
        except Exception as exc:
            raise RuntimeError(
                f'OpenVINO {device} 后端加载失败；当前默认设备为 {device}，'
                '请确认驱动和设备可用，或显式设置 REX_DEVICE=CPU 回退。'
            ) from exc
        self.DEVICE = torch.device('cpu')
        self.model = None
        self.inference = OpenVinoRexInference(self.rex_tokenizer, self.compiled_model)
        self.audit = dict(model='RexUniNLU-base', backend='openvino',
                          device=device, inference_threads=self.threads if device == 'CPU' else None,
                          openvino_path=str(ov_path), adapter_loaded=False,
                          adapter_parameter_count=0, missing_keys=[], unexpected_keys=[])

    def _load_pytorch(self, model_path):
        checkpoint = model_path / 'pytorch_model.bin'
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Rex checkpoint not found: {checkpoint}')
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        config.vocab_size = len(self.rex_tokenizer)
        self.model = RexModel(config)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(f'Base checkpoint missing keys: {incompatible.missing_keys}')
        self.model.to(self.DEVICE).eval()
        self.inference = RexInference(self.rex_tokenizer, self.model, self.DEVICE)
        self.audit = dict(model='RexUniNLU-base', backend='pytorch',
                          device=str(self.DEVICE), inference_threads=torch.get_num_threads(),
                          base_path=str(model_path), adapter_loaded=False,
                          adapter_parameter_count=0,
                          missing_keys=list(incompatible.missing_keys),
                          unexpected_keys=list(incompatible.unexpected_keys))

    def run_rex(self, text, config):
        """对一个正文窗口执行 Rex 推理。"""
        labels = [label for label, _ in config['schema']]
        if self.backend == 'pytorch':
            with torch.inference_mode():
                return {'entities': self.inference.extract(text, labels)}
        return {'entities': self.inference.extract(text, labels)}
