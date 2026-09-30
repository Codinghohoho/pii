"""修改模型目录、schema 和窗口大小，只需要看这个文件。"""
import os
from pathlib import Path

# 默认从本文件旁边的 model/rex-base 读取，换工作目录也不会失效。
# 只有显式设置 REX_MODEL_PATH 时才使用外部模型目录。
# 该目录包含权重、网络配置和 tokenizer 文件；推理网络代码在本模块 rex_network/。
MODULE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ['REX_MODEL_PATH']).expanduser() if os.environ.get('REX_MODEL_PATH') \
    else MODULE_DIR / 'model' / 'rex-base'

# 默认使用已转换的 OpenVINO GPU 模型；普通调用不需要设置环境变量。
# 需要排查转换模型时可临时设置 REX_BACKEND=pytorch 或 REX_DEVICE=CPU。
REX_BACKEND = os.environ.get('REX_BACKEND', 'openvino').lower()
REX_DEVICE = os.environ.get('REX_DEVICE', 'GPU').upper()
REX_THREADS = int(os.environ.get('REX_THREADS', '2'))
OPENVINO_MODEL_PATH = Path(os.environ['REX_OPENVINO_PATH']).expanduser() \
    if os.environ.get('REX_OPENVINO_PATH') \
    else MODEL_PATH.parent / 'rex-openvino' / 'rex.xml'

# 这是“程序内部的类型编号”，不是给用户看的结果文字。
# 左侧英文便于代码判断，右侧中文用于页面说明和日志。
# 规则可以直接输出 PATIENT/DOCTOR/CONTACT；Rex 的五种姓名问题统一转成 PERSON。
REX_SCHEMA = {
    '患者姓名': 'PERSON', '医生姓名': 'PERSON', '联系人姓名': 'PERSON',
    '亲属姓名': 'PERSON', '护士姓名': 'PERSON',
    '家庭住址': 'ADDRESS', '医院名称': 'ORG',
}
TEXT_TOKENS = 300      # 单个正文窗口的 token 上限，不是字符数。
OVERLAP_TOKENS = 48    # 相邻窗口复用的 token 数，减少实体被窗口边界切断的影响。
MODEL_TOKENS = 512    # 正文 + schema + 特殊 token 的总预算上限。

# 规则或 Rex 最终得到的内部类型 -> 中文显示名称。
# 这里只是标签表，不会自动产生识别能力，也不会决定哪些字段被识别。
FIELDS = {
    'PATIENT': '患者姓名', 'DOCTOR': '医生姓名', 'CONTACT': '联系人姓名',
    'PERSON': '人名', 'ID': '身份证号', 'PHONE': '联系电话',
    'ADDRESS': '详细住址', 'EMAIL': '邮箱', 'MRN': '病历/住院号',
    'EXAM': '检查申请号', 'BED': '床号', 'ORG': '医院名称',
    'DATE': '具体日期', 'AGE': '年龄',
}
# 占位符复用分组：各种姓名共享 PERSON 的映射和编号，不区分同名人物。
# 仅在生成占位符时归组，识别候选和 applied 中仍保留原始类型。
TOKEN_GROUPS = {'PATIENT': 'PERSON', 'DOCTOR': 'PERSON', 'CONTACT': 'PERSON'}

# 内部类型/复用分组 -> 脱敏占位符中的中文名称。
# 姓名类型先按 TOKEN_GROUPS 归为 PERSON，生成【人名1】；AGE 生成【年龄1】。
# 患者/医生/联系人旧标签保留，用于拒绝包含旧格式占位符的输入。
# 这张表只影响替换文字，不改变规则或 Rex 的识别范围。
# 这里完整列出每一项，避免用 **FIELDS 隐藏“年龄”等继承字段。
TOKEN_LABELS = {
    'PATIENT': '患者',
    'DOCTOR': '医生',
    'CONTACT': '联系人',
    'PERSON': '人名',
    'ID': '身份证',
    'PHONE': '电话',
    'ADDRESS': '地址',
    'EMAIL': '邮箱',
    'MRN': '病历号',
    'EXAM': '检查号',
    'BED': '床号',
    'ORG': '医院',
    'DATE': '日期',
    'AGE': '年龄',
}
