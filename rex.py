"""加载一次模型；长文本按 token 分窗，将局部位置换算回全文位置。"""
from .config import MODEL_PATH, MODEL_TOKENS, TEXT_TOKENS, OVERLAP_TOKENS, REX_SCHEMA
from .rules import make_span


class RexExtractor:
    """保存一个模型运行实例，负责推理窗口的组织，不负责实体重叠合并。"""
    def __init__(self, model_path=MODEL_PATH):
        """创建长文本抽取器。

        输入：``model_path``，本地 Rex 模型目录。
        输出：无；模型延迟到第一次 ``detect`` 调用时加载。
        """
        # 这里只存路径；第一次 detect() 时才占用内存/显存加载模型。
        self.model_path = model_path
        self.runtime = None

    def detect(self, text):
        """分窗处理全文并返回 Rex 候选。

        输入：``text``，待识别的完整原文。
        输出：``(spans, window_info)``；``spans`` 是候选实体列表，坐标对应全文；
        ``window_info`` 包含窗口范围、token 数、实际模型调用次数等统计。
        """
        # 与规则独立读取原文：姓名、地址等规则的冒号条件不限制 Rex。
        if self.runtime is None:
            from .model_loader import RexBaseRuntime
            self.runtime = RexBaseRuntime(self.model_path)
        runtime = self.runtime
        tokenizer = runtime.rex_tokenizer

        # schema 也占 token。给正文预留空间，避免超过模型的 512 token。
        hints = runtime.inference.encode_schema(REX_SCHEMA)[1]
        schema_tokens = max(len(tokenizer.encode(h, add_special_tokens=False)) for h in hints)
        # 16 是额外预留量；schema 变长时正文窗口会缩小，而不是截掉原文。
        budget = min(TEXT_TOKENS, MODEL_TOKENS - schema_tokens
                     - tokenizer.num_special_tokens_to_add(pair=True) - 16)
        if budget <= max(64, OVERLAP_TOKENS):
            raise ValueError('schema 太长，正文窗口不足')

        # offsets[i] 是第 i 个 token 在原文里的字符起止位置，和换行无关。
        offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True)['offset_mapping']
        spans, windows, cache = [], [], {}
        # first/last 是 token 下标；start/end 是字符位置，不能混用。
        first = 0
        while first < len(offsets):
            last = min(first + budget, len(offsets))
            start, end = offsets[first][0], offsets[last - 1][1]
            if end <= start:
                raise ValueError('tokenizer 返回了无效的窗口位置')
            segment = text[start:end]
            windows.append(dict(start=start, end=end, tokens=last-first))
            if segment not in cache:
                # 只在当前批次缓存完全相同的窗口，返回后不会保留这些正文片段。
                cache[segment] = runtime.run_rex(segment, {'schema': list(REX_SCHEMA.items())})
            for entity in cache[segment]['entities']:
                # Rex 返回 schema 标签；先转成模块使用的内部类型。
                kind = REX_SCHEMA.get(entity['label'])
                offset = entity['offset']
                if not kind or len(offset) != 2:
                    continue
                a, b = offset
                # 除了检查坐标合法，还核对该范围的文字确实等于模型报告的实体。
                if (isinstance(a, int) and isinstance(b, int) and 0 <= a < b <= len(segment)
                        and segment[a:b] == entity['text']):
                    # 窗口中的位置 + 窗口起点 = 全文中的位置。
                    spans.append(make_span(text, start+a, start+b, kind, 'Rex'))
            if last == len(offsets):
                break
            # 例如窗口 [0,300)，下一窗口从 token 252 开始，重叠最后 48 个。
            # 重叠区可能重复识别同一实体，交给 pipeline.merge_spans() 去重。
            first = last - OVERLAP_TOKENS

        # windows 是窗口记录；model_calls 可能因重复窗口复用而少于窗口数。
        return spans, dict(windows=windows, model_calls=len(cache), text_tokens=len(offsets),
                           text_budget=budget, schema_tokens=schema_tokens,
                           overlap_tokens=OVERLAP_TOKENS)
