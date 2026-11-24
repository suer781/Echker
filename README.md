# Qwen 双半球持续在线学习助手

主线模型为 `Qwen/Qwen2.5-0.5B-Instruct`。它使用 Qwen 词表和因果语言模型目标；旧字节级模型仍保留为历史实现，不作为 Qwen 迁移完成的验收依据。

## 运行

```powershell
.\.venv-life\Scripts\python.exe -m pip install -r .\requirements.txt
.\start_agent.ps1
```

也可以直接运行 `.\.venv-life\Scripts\python.exe .\chat_qwen.py`。可通过 `--model` 指定 Hugging Face 模型或本地模型目录，通过 `--state-dir` 指定双半球检查点目录。`--device cpu` 可在 CUDA 不可用或显存不足时运行，但全参数反向传播会明显变慢。

## 在线学习路径

- 左右半球分别保存独立完整 Qwen 权重、优化器状态、回放和债务账本；沿用现有 ByT5 双半球的 CPU/GPU 顺序驻留机制，GPU 同时只装载一个半球。
- 回复按 8 个新 token 分块生成。每块生成后立即对活动半球执行一次全参数 Qwen 因果损失更新，下一块使用更新后的权重继续生成。
- 完成的对话以 Qwen chat template 编码，所有角色的 token 都进入训练目标和持久回放；被动半球在 CPU 上后台巩固，不卸载 GPU 上负责服务的活动半球。巩固完成后再等待当前回复结束，并按债务压力决定是否切换半球。
- MLP 结构控制器根据损失变化和神经元重要性，扩展或裁剪 Qwen 各层的 `gate_proj`、`up_proj`、`down_proj` 宽度；动态宽度与激活统计随半球检查点保存。
- 外部工具消息可通过 `QwenAgent.learn_from_transcript(messages)` 传入；仓库当前 CLI 没有工具执行器，因此不会声称它能自动采集未接入的工具调用或结果。
- `run_agent.py` 和 `start_agent.ps1` 已指向 Qwen 主线；旧字节数据集/ByT5 看门狗入口已禁用，避免误启动归档训练链路。
- 学习允许新经验覆盖旧能力，不承诺避免灾难性遗忘；HTIM 是后续记忆机制的参考方向，本实现尚未宣称复现 HTIM。

CPU 巩固会与 GPU 推理并行争用系统内存带宽和处理器时间，低配设备上的响应可能变慢；正常请求不再等待整段被动巩固完成。进程关闭或显式保存时仍会等待后台巩固结束并保存检查点。

所有角色自动作为训练目标意味着用户消息、助手输出以及传入的工具调用/结果都会被模型模仿；错误或不可信内容也可能被学到。请勿传入密码、访问令牌等敏感信息。

## 验收边界

`tests/test_qwen_agent.py` 和内存测试验证的是代码级标签、回放与账本行为，不代表已验证实际 Qwen 权重上的学习质量。GTX 1060 3GB 环境需要实测完整模型生成、全参数反向、结构变更和恢复；遇到 CUDA OOM 时不要把跳过训练当作迁移成功。

## 主线文件

- [chat_qwen.py](chat_qwen.py)：交互 CLI。
- [qwen_agent.py](qwen_agent.py)：分块推理、TTT 和全角色 transcript 接口。
- [qwen_dual.py](qwen_dual.py)：Qwen 半球、动态 MLP 结构与现有双半球状态管理的衔接。
- [life_agent/memory.py](life_agent/memory.py)：共享回放/债务账本，支持字节与词表 token 两种统计域。
