"""规则与 Rex 识别 → 历史映射补匹配 → Presidio 脱敏 → 精确回填。"""
import re
import threading
import phonenumbers
from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, ConflictResolutionStrategy
from .config import MODEL_PATH, TOKEN_GROUPS, TOKEN_LABELS, FIELDS
from .rules import detect_rules
from .rex import RexExtractor

TOKEN_PATTERN = re.compile(
    r'【(?:' + '|'.join(map(re.escape, TOKEN_LABELS.values())) + r')\d+】'
)


def to_result(span):
    """把内部候选转换为 Presidio 结果。

    输入：``span``，包含 ``start/end/type/source/text`` 的候选字典。
    输出：``RecognizerResult``，供 Presidio 冲突处理和替换使用。
    分数表示处理优先级，不是模型概率。
    """
    # Rex 与规则谁优先由 merge_spans() 决定，不依赖这里的分数竞争。
    # 这些分数用于交给 Presidio 后的剩余冲突处理。
    score = 0.60 if span['source'] == 'Rex' else 0.80
    if span['source'] != 'Rex' and span['type'] == 'ID':
        score = 0.95 if '通过；未验证' in span['detail'] else 0.75
    if span['source'] != 'Rex' and span['type'] == 'PHONE':
        # CN 是未提供国家区号时的默认地区。校验失败降低分数，但不丢弃号码。
        try:
            valid = phonenumbers.is_valid_number(phonenumbers.parse(span['text'], 'CN'))
        except phonenumbers.NumberParseException:
            valid = False
        score = 0.90 if valid else 0.75
    return RecognizerResult(span['type'], span['start'], span['end'], score)


def merge_spans(text, rules, rex):
    """合并规则候选和 Rex 候选。

    输入：``text`` 原文，``rules`` 规则候选列表，``rex`` Rex 候选列表。
    输出：``(kept, conflicts)``；``kept`` 是保留的 Presidio 范围，
    ``conflicts`` 是规则覆盖 Rex 且边界不同的冲突记录。

    例如规则是“张三”，Rex 是“张三丰”：保留“张三”，记录整条 Rex 冲突。
    “丰”不会在此处被自动补上；调用方可用冲突信息检查是否需要人工处理。
    """
    # 先分别去重，避免分窗重叠产生的同一候选被反复展示。
    rules = EntityRecognizer.remove_duplicates([to_result(s) for s in rules])
    rex = EntityRecognizer.remove_duplicates([to_result(s) for s in rex])
    kept, conflicts = list(rules), []
    describe = lambda s: dict(type=s.entity_type, start=s.start, end=s.end,
                              text=text[s.start:s.end])
    for model in rex:
        # 两个左闭右开区间有交集才冲突；[0,2) 与 [2,4) 只是相邻。
        overlaps = [r for r in rules if model.start < r.end and r.start < model.end]
        if not overlaps:
            kept.append(model)
        # 有重叠的 Rex 不进入 kept。边界不同才记录冲突，边界相同无需重复提示。
        elif len(overlaps) != 1 or (model.start, model.end) != (overlaps[0].start, overlaps[0].end):
            conflicts.append(dict(rex=describe(model), rules=[describe(r) for r in overlaps]))
    return EntityRecognizer.remove_duplicates(kept), conflicts


def match_known_spans(text, lookup, detected):
    """在本轮识别未覆盖的位置，按历史原文补充候选。

    不改写原文，保持字符坐标；同一原文属于多个复用分组时不猜类型。
    历史候选按长度优先，避免短姓名抢占已知长姓名的一部分。
    床号、年龄、单字符和不足六位的纯数字不做无标签传播。
    """
    groups_by_value = {}
    for group, value in lookup:
        groups_by_value.setdefault(value, set()).add(group)
    known = {}
    for value, groups in groups_by_value.items():
        if len(groups) != 1 or len(value) < 2:
            continue
        group = next(iter(groups))
        if group in {'BED', 'AGE'} or (value.isdecimal() and len(value) < 6):
            continue
        known[value] = group
    if not known:
        return []

    patterns = []
    for value in sorted(known, key=lambda value: (-len(value), value)):
        pattern = re.escape(value)
        # 英文/数字值不能截取更长编号的一部分；中文上下文仍可直接邻接。
        if value[0].isascii() and (value[0].isalnum() or value[0] == '_'):
            pattern = r'(?<![A-Za-z0-9_])' + pattern
        if value[-1].isascii() and (value[-1].isalnum() or value[-1] == '_'):
            pattern += r'(?![A-Za-z0-9_])'
        patterns.append(pattern)

    # 前瞻允许收集起点不同但相互重叠的候选，再统一选较长值。
    expression = r'(?=(?P<value>' + '|'.join(patterns) + r'))'
    candidates = []
    for match in re.finditer(expression, text):
        start, end = match.span('value')
        if any(start < span.end and span.start < end for span in detected):
            continue
        value = match.group('value')
        candidates.append(RecognizerResult(known[value], start, end, 0.8))

    kept = []
    for candidate in sorted(candidates, key=lambda span: (span.start - span.end, span.start)):
        if not any(candidate.start < span.end and span.start < candidate.end for span in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda span: span.start)


class RedactionSession:
    """一个实例对应一份文档或一个会话：复用 Rex，跨批次共享映射。"""
    def __init__(self, model_path=MODEL_PATH):
        """创建独立脱敏会话。

        输入：``model_path``，本地 Rex 模型目录。
        输出：无；会话保存模型、占位符映射和编号状态。
        """
        self.rex = RexExtractor(model_path)
        self.mapping = {}              # 占位符 -> 原始文字
        self.lookup = {}               # (复用分组, 原始文字) -> 占位符；姓名统一为 PERSON
        self.counts = {}               # 每个复用分组独立编号
        self.lock = threading.Lock()   # 同一实例按顺序处理，避免编号和模型状态竞争。

    def redact(self, text):
        """脱敏一批文本，并把本批成功生成的映射加入当前会话。

        输入：``text``，非空且尚未脱敏的原始文本。
        输出：字典，包含 ``redacted``（脱敏正文）、``mapping``（会话映射副本）、
        ``applied``（实际替换范围）、
        conflicts（规则与 Rex 范围冲突）及 windows（窗口统计）。
        调用方应只把 redacted 送给后续大模型，mapping 留在本地用于恢复。
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError('请输入非空文本')
        with self.lock:
            # 不把已有占位符当作新原文，否则无法保证精确恢复。
            if TOKEN_PATTERN.search(text):
                raise ValueError('输入包含脱敏占位符，请传入原始文本')
            # 第一步：规则扫描整批文本；Rex 自己分窗。这里不按行拆分字段。
            rules = detect_rules(text)
            rex, window_info = self.rex.detect(text)
            # 第二步：按规则优先筛选，不改变已识别的规则边界。
            selected, conflicts = merge_spans(text, rules, rex)
            # 已登记原文无需本轮再次命中识别器；只补未覆盖的位置，
            # 避免把本轮识别出的完整姓名/字段截成历史短值。
            selected.extend(match_known_spans(text, self.lookup, selected))
            # 临时映射：只有本批成功完成并通过回填校验后才提交到会话。
            mapping, lookup, counts = dict(self.mapping), dict(self.lookup), dict(self.counts)

            def operator(kind):
                """为每种类型生成 Presidio 替换回调，用闭包记住对应的 kind。"""
                group = TOKEN_GROUPS.get(kind, kind)
                def replace(value):
                    # 各姓名类型中相同原文复用同一个人名标记，不区分同名人物。
                    # 其他实体仍按各自类型复用；不改写原文，以保证精确回填。
                    key = (group, value)
                    if key not in lookup:
                        counts[group] = counts.get(group, 0) + 1
                        token = f'【{TOKEN_LABELS[group]}{counts[group]}】'
                        lookup[key] = token
                        mapping[token] = value
                    return lookup[key]
                return OperatorConfig('custom', {'lambda': replace})

            # 第三步：Presidio 处理剩余交集，并调用 replace(value) 生成占位符。
            # 不合并仅由空格分隔的实体；编号按回调分配，不保证原文从左到右顺序。
            output = AnonymizerEngine().anonymize(
                text=text, analyzer_results=selected,
                operators={kind: operator(kind) for kind in TOKEN_LABELS},
                conflict_resolution=ConflictResolutionStrategy.REMOVE_INTERSECTIONS,
                merge_entities_with_spaces=False,
            )
            # Presidio 返回的是替换后的位置，累计长度差可换回原文位置。
            # 例如 2 字姓名被 5 字占位符替换，后续坐标的 delta 要减 3。
            applied, delta = [], 0
            for item in sorted(output.items, key=lambda item: item.start):
                value = mapping[item.text]
                start = item.start + delta
                end = start + len(value)
                if text[start:end] != value:
                    raise RuntimeError('替换位置与原文不一致')
                applied.append(dict(type=item.entity_type,
                                    label=FIELDS.get(item.entity_type, item.entity_type),
                                    start=start, end=end, text=value,
                                    token=item.text))
                delta += len(value) - (item.end-item.start)
            # 第四步：确认“脱敏 -> 回填”与本批输入逐字相同，才提交会话状态。
            # 这里核对的是输入文本，不代表已纠正 OCR 内容或保证没有漏识别。
            if restore_exact(output.text, mapping) != text:
                raise RuntimeError('回填校验失败')
            self.mapping, self.lookup, self.counts = mapping, lookup, counts
            # 返回副本，后续批次增加映射不会改变之前调用拿到的 mapping。
            return dict(redacted=output.text, mapping=dict(mapping), applied=applied,
                        conflicts=conflicts, windows=window_info)

    def restore(self, text):
        """恢复占位符。

        输入：``text``，含占位符的文本。
        输出：字符串；只替换当前会话中完整且已知的占位符。
        """
        return restore_exact(text, self.mapping)


def restore_exact(text, mapping):
    """用映射精确回填占位符。

    输入：``text`` 含占位符的文本；``mapping`` 为“占位符 -> 原文”的字典。
    输出：恢复后的字符串；未知、被改写或被删掉的标记保持原样。
    """
    if not mapping:
        return text
    # escape 让占位符按字面匹配；长标记优先，避免通用映射中短键抢先匹配。
    pattern = '|'.join(re.escape(k) for k in sorted(mapping, key=len, reverse=True))
    # 一次 re.sub 完成替换，不再次扫描回填出的文字，避免递归替换原文。
    return re.sub(pattern, lambda match: mapping[match[0]], text)
