# Qwen Chat (vLLM)

一个轻量级的聊天 Web Demo：使用 FastAPI 作为代理层，对接外部 vLLM 推理服务，前端为纯静态页面，实现流式输出、思考模式、多会话管理与本地持久化。

项目目标是验证 **vLLM + Web UI** 的最小可用部署方式，代码结构尽量简单，便于二次修改或集成到其他系统中。

---

## ⚙️ 运行环境

- Python ≥ 3.9（推荐 3.10）
- GPU 环境（CUDA 可用）
- Linux / AutoDL / 云服务器环境
- 安装依赖：

```bash
pip install -r requirements.txt
```

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

模型下载完成后，`cache_dir` 即为模型所在路径。

---

## 🔧 快速开始

### 1️⃣ 启动 vLLM（示例）

```bash
vllm serve /path/to/Qwen3-8B \     #这里换成你本地模型目录。
  --host 0.0.0.0 --port 7000 \     #如端口被占用可改其它端口，后端 `REMOTE_VLLM_BASE_URL` 也要同步。
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \    #根据显存余量调整，显存紧张就调低。
  --max-model-len 8192    #按显存与上下文需求调整。
```

- 多卡可使用 `--tensor-parallel-size N`；高吞吐场景可进一步调节 batch / eager 相关参数。
- 若需要思考模式解析（如 deepseek_r1 风格），可加 `--reasoning-parser deepseek_r1`。
- 健康检查：`curl http://<vllm-host>:7000/health` 应返回 200。

### 2️⃣ 启动代理后端（FastAPI）

```bash
cd qwen_chat
export WEB_HOST=0.0.0.0
export WEB_PORT=6006
export REMOTE_VLLM_BASE_URL=http://127.0.0.1:7000   # 改成你的 vLLM 地址
export REMOTE_VLLM_MODEL=Qwen3-8B                   # 与 vLLM 模型一致
export CHAT_SESSIONS_FILE=$(pwd)/chat_sessions.json  # 会话持久化文件
python web_server.py
```

### 3️⃣ 打开前端  

浏览器访问 `http://localhost:6006`（远程请替换主机 IP），可勾选思考模式查看 reasoning；多会话可自动/手动命名，历史落盘到 `chat_sessions.json`。



---

## 💡 提示
- 如需在代码目录外运行，可将 `STATIC_DIR` 或 `CHAT_SESSIONS_FILE` 配成绝对路径，确保静态资源与持久化文件可达。
