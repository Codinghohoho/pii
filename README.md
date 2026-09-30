# PII 文本脱敏

使用本地 Rex 模型和正则规则识别姓名、地址、电话、身份证等敏感信息，替换为占位符，并按会话映射回填原文。

**当前提供 Python 调用和命令行示例，尚无 HTTP API、会话持久化或 Hermes 接入。** Hermes 接入方案见 [HERMES.md](HERMES.md)。

默认使用 **OpenVINO + Intel GPU**，无需配置环境变量。第一次脱敏时加载、编译模型，之后在同一实例内复用。安装和排查见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 快速运行

在当前 `pii` 目录执行，使用已安装的 `.venv` 和项目模型。运行内置示例：

~~~bash
PYTHONPATH=.. .venv/bin/python -m pii.demo
~~~

或读取 UTF-8 文本文件：

~~~bash
PYTHONPATH=.. .venv/bin/python -m pii.demo --input input.txt
~~~

输出包含脱敏文本和回填校验结果。`PYTHONPATH=..` 用于找到 `pii` 包，不是 GPU 配置；`input.txt` 相对当前目录读取。

## Python 调用

同一会话复用一个 `RedactionSession`，独立会话创建独立实例：

~~~python
from pii.pipeline import RedactionSession

session = RedactionSession()
text = "患者姓名：张三，联系电话：13800138000"
result = session.redact(text)

print(result["redacted"])
print(session.restore(result["redacted"]))

# 继续使用同一实例：已识别的“张三”复用旧标记，新增“李四”分配新标记。
second = session.redact("请查张三的记录。联系人：李四")
print(second["redacted"])
~~~

## 会话映射

- 姓名统一使用 `【人名N】`，同一原文复用同一个标记，不区分同名人物；其他实体按“类型 + 原文”复用。
- 已登记的原文再次出现时，会在本轮识别未覆盖的位置直接匹配并复用旧标记，无需字段标签。床号、年龄、单字符、短纯数字及类型歧义值不做无标签匹配。
- 映射只保存在当前实例内存中。新建实例或进程重启后不会恢复；不同实例也不共享模型。

## 返回结果

`session.redact(text)` 返回：

| 键 | 内容 |
|---|---|
| `redacted` | 脱敏文本，供后续模型处理 |
| `mapping` | 当前会话“占位符 → 原文”的副本 |
| `applied` | 本批实际替换的类型、原文、字符范围和 token |
| `conflicts` | 规则与 Rex 识别范围的冲突记录 |
| `windows` | 文本分窗及模型调用统计 |

**只把 `redacted` 发给后续大模型。** `mapping`、`applied`、`conflicts` 含敏感原文，应留在本地。字符范围使用原文坐标 `[start, end)`。

## 使用限制

- `redact()` 只接受非空原文，包含脱敏占位符时会报错，暂不支持原文与占位符混合输入。
- `restore()` 只恢复当前会话已知的完整标记；未知或被改写的标记保持原样。
- 回填一致不代表敏感信息全部识别；规则与模型冲突时规则优先，仍可能漏脱敏。
- 只处理文本，不包含 PDF 解析、OCR、语音或图像遮挡。
