# 蓝蓟智能（Echker）—— 自学习 AI 系统

开发代号：haolie（昊冽）

一个基于 PyTorch 与 Python 标准库自研实现的自学习 AI 系统，无第三方 LLM 框架依赖（无 nanoGPT / minGPT / HuggingFace / 现成 tokenizer）。模型直接学习原始字节流（律 L1），通过"无模式统一自循环"持续从输入中学习。

完整设计与生物学依据见 [架构设计.md](架构设计.md)。

## 核心机制

- **无模式统一自循环**：任何输入（用户对话、批量文件、数据流）统一成为学习信号，走同一条"经验 → 缓冲 → 选拔 → 睡眠训练 → 体检门控 → 换班"链路。
- **无限循环喂食**：数据源喂尽后自动轮转回第一个源；间隔重复作为巩固机制。
- **杏仁核托管**：生成温度 / top_k / top_p 由 Amygdala 根据系统健康度自动调节，无需人工设置。
- **体检门控**：固定探测集（任何 hemisphere 永不在此训练）作为换班金丝雀，通过才上岗。

## 安装

```bash
pip install -r requirements.txt
```

依赖：`torch>=2.0`（见 [requirements.txt](requirements.txt)）。

## 快速开始

```bash
python chat.py                    # 聊天入口（输入自动成为学习信号，后台睡眠训练保持自循环）
python feed.py --resume           # 从 dolphin/fed_state.pt 按喂食游标续喂（启动自循环）
python pretrain.py --steps 300   # 从 corpus/ 干净语料预训练基座
python smoke_test.py              # M0 端到端冒烟测试
```

## 存档与持久化

- 系统运行中每 5 个睡眠周期自动存档到 `dolphin/fed_state.pt`（原子写 + `dolphin/archive/` 滚动备份 3 份），退出（Ctrl+C/收工）时也会存档。
- 下次启动 `python feed.py --resume` 或 `python chat.py` 会自动读档续跑（恢复权重、优化器动量、记忆、喂食游标等全部状态）；无存档则随机初始化，从零积累。

## 运行测试

```bash
python tests/test_conformance.py && python tests/test_growth_support.py && python tests/test_mean_nll_batch.py
```

或使用 pytest（`pytest.ini` 已配置 `t_*` 收集规则）：

```bash
pytest tests/ -q
```

## 项目结构

| 路径 | 职责 |
|---|---|
| `dolphin/model.py` | 字节级因果 Transformer（自研，无 tokenizer） |
| `dolphin/experience.py` | 海马体：带通价值门控、窗口去重惩罚、睡眠压力 |
| `dolphin/memory.py` | 记忆库：滞留、检索（预支）、复活（间隔重复） |
| `dolphin/life.py` | 睡眠周期全流程与结构恒温器（律 L5–L11） |
| `dolphin/dolphin.py` | 双半球管理、服务接口、持久化 |
| `dolphin/amygdala.py` | 杏仁核：生成参数自动托管 |
| `feed.py` | 部署期喂食驱动器（守护模式、无限循环） |
| `chat.py` | 聊天入口 |
| `probes/` | 固定探测集与体检门控 |
| `tests/` | 一致性、成长支持、均值 NLL 批次测试 |
| `corpus/` | 胎教语料（只放干净来源） |
| `实验记录/` | 实验报告与归档 |

## 律与值

律（不可违反）与值（系统自理的反馈量）见《架构设计.md》第 1、2 节。
一句话版本：**律固定，值自成。**