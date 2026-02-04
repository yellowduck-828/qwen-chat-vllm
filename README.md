# Qwen Chat (vLLM)

一个基于 **vLLM + Qwen 模型** 的轻量级对话服务示例，提供简单的 Web 界面与后端推理服务，
主要用于验证 **大模型推理部署、服务化与工程流程**。

本项目聚焦于：
- vLLM 推理服务的最小可用部署
- Web API 与前端交互
- 工程结构与可复现运行方式

---

## ⚙️ 运行环境

- Python ≥ 3.9（推荐 3.10）
- GPU 环境（CUDA 可用）
- Linux / AutoDL / 云服务器环境

---

## 📦 模型下载

本项目 **不包含任何模型权重文件**，请在运行前自行下载所需的 Qwen 模型，并在代码中配置对应路径。

推荐使用 ModelScope 提供的 `snapshot_download` 接口进行模型下载。

新建一个脚本文件，例如 `model_download.py`，并写入以下内容：

```python
from modelscope import snapshot_download

model_dir = snapshot_download(
    'Qwen/Qwen3-8B',
    cache_dir='  ',      #填入你自己的模型下载路径
    revision='master'
)
```

然后在终端中执行：

```python
python model_download.py
```

模型下载完成后，` cache_dir` 即为模型所在路径。
