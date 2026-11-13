# 并发审查原始记录（concurrency audit）

审查对象：`life_agent/agent.py`、`life_agent/worker.py`（只读，未修改任何源码/测试/验证产物）。
运行环境：Windows，Python 3.13.18，torch 2.8.0+cu126，GPU NVIDIA GeForce GTX 1060 3GB（CUDA 可用，未 OOM）。
解释器：`.venv-life\Scripts\python.exe`；运行时 `PYTHONDONTWRITEBYTECODE=1`，未在 `life_agent/`、`tests/`、`validation_runs/`、`state_dataset_smoke/` 生成 `__pycache__` 或 `.pytest_cache`。

## 隔离配置（`config_audit.json` / `config_audit2.json`）

- d_model 64、n_heads 4、n_layers 2、hidden_dim 128、min 64、max 256、max_seq_len 128。
- batch_size 4、consolidation_steps 1、feed_steps 1、replay/debt capacity 128。
- 结构神经发生关闭（structure_interval/warmup = 100000），排除 resize 干扰。
- audit1：checkpoint_interval_s=99999；audit2：checkpoint_interval_s=2.0（触发 monitor 保存）。

## 执行命令

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
# 实验 1：64 轮 feed/interact 混合流
& ".venv-life\Scripts\python.exe" "_audit_tmp\concurrency\run_audit.py"
# 分析
& ".venv-life\Scripts\python.exe" "_audit_tmp\concurrency\analyze_audit.py"
& ".venv-life\Scripts\python.exe" "_audit_tmp\concurrency\inspect_switches.py"
# 实验 2：24 轮混合 + 36 轮纯 interact，2s monitor 检查点
& ".venv-life\Scripts\python.exe" "_audit_tmp\concurrency\run_audit2.py"
& ".venv-life\Scripts\python.exe" "_audit_tmp\concurrency\inspect_audit2.py"
& ".venv-life\Scripts\python.exe" "_audit_tmp\concurrency\check_pending.py"
```

## 实验 1 结果摘录（audit_metrics.json / audit_summary.json）

- 64 轮（feed 32 / interact 32），切换 18 次；最终步数 left=114、right=120。
- job 计数：train 32、consolidate 84、sync 18、ttt 118、snapshot 1；wait=False future 66 个全部 resolve，异常 0。
- wait=True 调用跨度：均值 15.7ms，p95 27.2ms，最大 286.1ms（1 个 train job）。
- compute 时长：train 均值 21.3ms/最大 285.9ms；consolidate 均值 13.9ms/最大 20.5ms；ttt 均值 12.6ms/最大 18.5ms。
- drain 记录 32 条，全部有 future，`step_delta` 全为 1；drain 均值 13.75ms、最大 20.6ms，无异常。
- schedule 86 次：新提交 future 66 次；延迟合并（pending 仍在跑，未提交新 job）20 次；按线程 MainThread 82、homeostasis 4。
- maybe_switch 68 次：
  - MainThread 64 次，early_return 0，切换 18；
  - homeostasis 4 次，early_return 4，切换 0（记录里残留的 `should_switch=true` 是上一次 MainThread decide 留下的 `last_decision`，monitor 提前返回前并未重新 decide）。
  - `early_return_after_monitor = []`：未观测到 monitor 改写探针后跳过 main 检查。
- 队列峰值 qsize=1；observer 线程轮询 `status()`：均值 40.6ms、最大 166.6ms，无异常。
- close 耗时 55.2ms，结束 qsize=0、pending={}，无异常。

### 切换后 pending 双残留摘录（inspect_switches.py）

```
cycle 11 interact active=right pending=["right","left"] steps L22/R25 switches=5
cycle 19 interact active=right pending=["right","left"] steps L37/R41 switches=7
cycle 47 interact active=left  pending=["left","right"] steps L86/R93 switches=14
cycle 63 interact active=left pending=["left","right"] steps L114/R119 switches=18
```
23/64 轮 pending 字典同时保留两个半球的 key。

## 实验 2 结果摘录（audit2_metrics.json / audit2_summary.json）

### monitor 保存时刻快照与待完成巩固

```
save homeostasis t=25708.034 pending_pending=["left"]
  snapshot: left step=32 debt=26 replay=9 ; right step=66 debt=9 replay=35
save homeostasis t=25710.367 pending_pending=["left"]
  snapshot: left step=74 debt=25 replay=15 ; right step=157 debt=15 replay=91
save MainThread  t=25710.479 pending_pending=[]   # close 路径，wait 之后保存，一致
  snapshot: left step=76 debt=23 replay=15 ; right step=159 debt=15 replay=93
```
两次 monitor 保存时 left 的 consolidate future 尚未完成（pending_done left=false）；snapshot 优先级 2 排在 consolidate 4 之前执行。

### 纯 interact 阶段（phase B，36 次 interact）

- consolidate 40、ttt 114、train 0、snapshot 3、sync 2；qsize 峰值 1。
- 步数增量 left=47、right=107（右为活动侧较多）；结束 pending=[right,left]。
- 本负载下未复现被动侧巩固饥饿；残余风险在更高频 ttt / 更慢硬件下尚未验证。

## 复核过的安全项（未见缺陷）

- 模型参数 / optimizer / `global_step` 仅 worker 线程写；`agent.py` 不直接写模型或优化器。
- replay/debt 为双写（主线程 `_remember` 入台账，worker 回写样本 weight/loss），变更均在对应 `state.lock` 内。
- CPU 推理读 `cpu_model` 持 `state.lock`；worker 发布 `cpu_model`（load_state_dict / resize）同样持锁；resize 先改 GPU 再改 CPU，推理不会读到半更新的 CPU 权重。
- snapshot 模型权重与 optimizer 状态经 `detach().cpu()` 拷贝（`_to_cpu`），未发现张量引用逃逸。
- `_synchronize` 在 worker 单线程内串行两侧 GPU 插值，CPU 推理不触碰 GPU 权重。
- 同优先级 tie-break 由 `sequence` 序号 FIFO 决定，确定无歧义；同半球 pending 在跑时新巩固被延迟合并，样本不丢弃（debt 台账保留）。

## 未覆盖 / 局限

- 未注入 worker 线程致命崩溃或挂起任务，future 永久 pending / close 挂起路径仅为代码推断。
- 未跑 10 GiB 全量训练；小模型、小批次下队列与临界区时长偏小，大负载下的放大倍数未测量。
- monitor 路径背景失败被吞的行为由代码路径推断，未用故障注入复现。
