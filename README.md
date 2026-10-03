# 海豚 Dolphin —— 自学习 AI（M0）

双半球轮换 + 睡眠周期 + 价值门控 + 部分有损蒸馏交接。

**全自研声明**：只依赖 PyTorch 与 Python 标准库；无 nanoGPT/minGPT/HuggingFace，
无任何现成 tokenizer（律 L1：模型直接吃原始字节流）。完整设计与生物学依据见
[架构设计.md](架构设计.md)。

## 快速开始

```bash
python pretrain.py --steps 300   # 胎教：从 corpus/ 干净语料预训练基座
python smoke_test.py             # M0 端到端冒烟测试
```

## 作为库使用

```python
from dolphin.dolphin import Dolphin

d = Dolphin.from_birth("dolphin/birth.pt")   # 从基座出生（双半球同源）
resp, eid = d.serve("人为什么要睡觉？")       # 醒脑服务（权重冻结）
d.feedback(eid, 1.0)                         # 用户奖励
d.maybe_sleep()                              # 压力够就睡：选拔→变异重放→蒸馏→体检→换班
d.save("dolphin/state.pt")                   # 存档
```

## 目录

| 路径 | 职责 |
|---|---|
| `dolphin/model.py` | 字节级因果 Transformer（自研） |
| `dolphin/experience.py` | 海马体：带通价值门控、窗口去重惩罚、睡眠压力 |
| `dolphin/sleep.py` | 睡眠周期全流程（律 L5-L10） |
| `dolphin/memory.py` | 记忆库：滞留、检索（预支）、复活（间隔重复） |
| `dolphin/probe.py` | 固定探测集（任何训练永不触碰） |
| `dolphin/dolphin.py` | 双半球管理、服务接口、持久化 |
| `corpus/` | 胎教语料（只放干净来源） |
| `probes/probe.txt` | 体检探测集 |

## 律与值

律（不可违反）与值（系统自理的反馈量）见《架构设计.md》第 1、2 节。
一句话版本：**律固定，值自成。**
