# Qwen 在线纠错学习助手

主线模型：`Qwen/Qwen2.5-0.5B-Instruct`。它使用常见的子词词表与因果语言模型接口，不是字节级 ByT5。

旧 ByT5 实现已保存在 `legacy-byte-level-frozen` 分支，并标记为永久停更。主线不再把旧实现的测试结果当作 Qwen 的验收证据。

## 开始使用

安装 [requirements.txt](requirements.txt) 中的依赖后运行：

```powershell
.\.venv-life\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv-life\Scripts\python.exe .\chat_qwen.py
```

首次运行会从 Hugging Face 下载模型。也可以通过 `--model` 指向兼容的本地模型目录或其他模型 ID。

普通对话**不会自动更新权重**。若回答有误，在下一条命令输入：

```text
/teach <你认可的正确回答>
```

系统才会用这条明确纠正进行一次 LoRA 更新，并保存 adapter、优化器状态和最多 128 条纠正样本；回放样本用于降低新纠正覆盖旧纠正的风险。`/status` 查看模型、设备和样本数，`/save` 保存，`/exit` 保存退出。

## 学习与验收边界

- 这是“根据明确纠正进行增量监督更新”，不是无监督地产生正确知识；没有可信反馈时，系统无法判断自己的答案是否正确。
- 代码只训练 LoRA adapter，不覆盖基础模型文件。模型权重和纠正数据保存在本地 `state_qwen/`，不会提交到仓库。
- 目前只实现对话文本；没有声称支持图片、音频或字节流训练。
- 本机 GTX 1060 3GB 尚未对 Qwen 生成和 LoRA 更新做实机显存验收。若 CUDA OOM，可用 `--device cpu`，但会明显变慢；不要在验证前启动长时间训练。
- `tests/test_qwen_agent.py` 只验证训练标签确实屏蔽提示词、只对纠正答案计算损失；它不证明模型回答质量或泛化提升。需要用固定、人工核验的测试集比较更新前后的结果后，才能宣称有改善。

## 主线文件

- [chat_qwen.py](chat_qwen.py)：交互命令行入口。
- [qwen_agent.py](qwen_agent.py)：Qwen 加载、对话、纠正样本学习和 adapter 持久化。
- [tests/test_qwen_agent.py](tests/test_qwen_agent.py)：训练目标掩码测试。
- `chat_dual.py`、`chat_cli.py`、`byt5_dual.py` 和旧 ByT5 训练/验证脚本是历史兼容代码，不再作为主线入口。
