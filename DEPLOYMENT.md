# 部署与排查

日常使用见 [README.md](README.md)。本页用于换机器安装、更换模型或排查运行问题；当前机器无需重复安装或转换模型。

## 首次安装

默认后端是 OpenVINO GPU，需要可被 OpenVINO 识别的 Intel GPU、对应驱动和设备访问权限。GPU 加载失败会报错，不会自动改用 CPU。

在当前 `pii` 目录创建新环境并安装依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

模型文件需另外准备，安装 Python 依赖不会下载权重：

| 目录 | 用途 |
|---|---|
| `model/rex-base/` | tokenizer、配置和原始 `pytorch_model.bin` |
| `model/rex-openvino/` | 转换后的 `rex.xml` 和 `rex.bin`，两者需一起部署 |

默认 OpenVINO 推理从 Rex-base 目录读取 tokenizer，从 OpenVINO 目录读取权重。第一次脱敏时编译模型，后续调用复用同一实例。

## 模型转换

仅在缺少 OpenVINO 模型或更换原始权重后执行：

```bash
PYTHONPATH=.. .venv/bin/python -m pii.convert_openvino
```

转换脚本读取原始 PyTorch 权重，输出 FP32 模型到项目内的 `model/rex-openvino/`。会覆盖该目录中的同名模型文件；更换权重后需重新验证识别结果。当前不加载 LoRA。

## 排查配置

以下是可选覆盖项，普通使用不需要设置。环境变量需在启动 Python 前设置，默认值见 [config.py](config.py)。

| 环境变量 | 默认值与作用 |
|---|---|
| `REX_BACKEND` | `openvino`；设置为 `pytorch` 使用原始权重 |
| `REX_DEVICE` | `GPU`；设置为 `CPU` 使用 OpenVINO CPU |
| `REX_THREADS` | `2`；只控制 OpenVINO CPU 推理线程数 |
| `REX_MODEL_PATH` | 项目内 `model/rex-base/`；tokenizer 和原始权重目录 |
| `REX_OPENVINO_PATH` | 默认在 Rex-base 目录旁的 `rex-openvino/rex.xml`；已转换模型路径 |

OpenVINO CPU 回退：

```bash
REX_DEVICE=CPU PYTHONPATH=.. .venv/bin/python -m pii.demo
```

PyTorch CPU 对照（当前机器的 PyTorch 是 CPU 版，线程在启动时限制为 2）：

```bash
REX_BACKEND=pytorch OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.. .venv/bin/python -m pii.demo
```

PyTorch 后端自动使用可用的 CUDA，否则使用 CPU；`REX_DEVICE` 和 `REX_THREADS` 不控制此后端。更换模型时，tokenizer 与 OpenVINO 权重必须来自同一套模型；默认模型路径不受工作目录影响。

如需查看替换范围、映射和冲突，可在 demo 命令末尾增加 `--details`。该输出包含敏感原文，不应发送给后续大模型。
