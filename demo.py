"""运行：python -m rules_rex_minimal.demo [--input 文件.txt] [--details]"""
import argparse
import json
from pathlib import Path
from .pipeline import RedactionSession


def main():
    """运行命令行示例。

    输入：命令行可选的 ``--input`` UTF-8 文件和 ``--details`` 开关。
    输出：打印脱敏文本、回填校验结果以及可选的详细 JSON；函数本身返回 ``None``。
    """
    parser = argparse.ArgumentParser(description='规则 + Rex 最小脱敏示例')
    parser.add_argument('--input', type=Path, help='UTF-8 原文文件；省略时运行虚构示例')
    parser.add_argument('--details', action='store_true', help='显示映射、范围、冲突及窗口信息')
    args = parser.parse_args()
    # utf-8-sig 同时兼容带 BOM 的 UTF-8 文件，避免把 BOM 当作正文字符。
    # 未指定文件时使用以下虚构示例，便于直接运行检查环境。
    text = (args.input.read_text(encoding='utf-8-sig') if args.input else
            '医院名称：春风医院\n患者姓名：张三\n医生：李四\n'
            '联系电话：13800138000\n住院号：ZY20260922001\n'
            '家庭住址：春风市和平路88号\n入院日期：2026年09月22日\n主诉：头痛三天。')
    session = RedactionSession()  # 多批输入时复用此实例，模型与映射也随之复用。
    result = session.redact(text)  # 模型在第一次调用时加载；长文本自动分窗。
    print(result['redacted'])
    print('\n精确回填校验：', session.restore(result['redacted']) == text)
    print('窗口数：', len(result['windows']['windows']), '范围冲突数：', len(result['conflicts']))
    if args.details:
        # 仅在显式要求时打印完整结构，其中 mapping 包含可恢复的原始敏感值。
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
