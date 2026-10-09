# Echker

自适应自循环学习系统。任何输入统一成为学习信号，经睡眠训练、体检门控、换班三个环节持续积累。

实验性研究项目，不构成生产环境可用性承诺。完整设计见 [架构设计.md](架构设计.md)。

## 核心机制

- 无模式自循环：任何输入（对话、批量文件、数据流）统一成为学习信号，走同一条训练链路。
- 无限循环喂食：数据源喂尽后自动轮转回第一个源，间隔重复作为巩固机制。
- 杏仁核托管：生成温度 / top_k / top_p 由系统根据自身健康度自动调节。
- 体检门控：在固定探测集上通过检查才换班，防止退化模型上岗。

## 安装

```bash
pip install -r requirements.txt
```

依赖：`torch>=2.0`（见 [requirements.txt](requirements.txt)）。

## 快速开始

```bash
# 聊天入口
python chat.py

# 从存档续跑自循环
python feed.py --resume

# 用 corpus/ 语料预训练基座
python pretrain.py --steps 300

# 端到端冒烟测试
python smoke_test.py
```

## 存档与持久化

- 每 5 个睡眠周期自动存档到 `dolphin/fed_state.pt`（原子写 + `dolphin/archive/` 滚动备份 3 份），退出时也会存档。
- `feed.py --resume` 与 `chat.py` 启动时自动读档；无存档则随机初始化。
- 存档文件（`dolphin/*.pt` 与 `dolphin/archive/`）已被 `.gitignore` 忽略，不入库。

## 运行测试

```bash
python tests/test_conformance.py && python tests/test_growth_support.py && python tests/test_mean_nll_batch.py
```

或使用 pytest：

```bash
pytest tests/ -q
```

## 项目结构

| 路径 | 职责 |
|---|---|
| `dolphin/model.py` | 字节级因果 Transformer |
| `dolphin/experience.py` | 海马体：带通价值门控、窗口去重惩罚、睡眠压力 |
| `dolphin/memory.py` | 记忆库：滞留、检索（预支）、复活（间隔重复） |
| `dolphin/life.py` | 睡眠周期全流程与结构恒温器 |
| `dolphin/dolphin.py` | 双半球管理、服务接口、持久化 |
| `dolphin/amygdala.py` | 杏仁核：生成参数自动托管 |
| `dolphin/autotune.py` | 值自成自动调参控制器 |
| `dolphin/param_adaptive.py` | 参数自适应控制律 |
| `feed.py` | 喂食驱动器（守护模式、无限循环） |
| `chat.py` | 聊天入口 |
| `probes/` | 固定探测集与体检门控 |
| `tests/` | 一致性、成长支持、均值 NLL 批次测试 |
| `corpus/` | 胎教语料 |

## 贡献指南

欢迎通过 GitHub Issue 提交 bug 报告或改进建议，通过 Pull Request 提交代码。提交前请保证 `tests/` 下三个测试脚本全部通过，核心机制改动请在 PR 描述中说明设计依据。

## 许可证

本项目以 [Apache License 2.0](LICENSE) 发布。