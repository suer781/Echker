"""金发姑娘价值门控（律 L4 补全）。

律 L4：「价值门控存在，且为带通：太熟跳过、太怪挂起、中间带优先」。
经验缓冲（experience.py）此前只实现了「中间带优先」的软排序（band()/select()
的高斯带通权），缺「太熟跳过」「太怪挂起」两个硬门控。本模块补全硬门控：
喂食入口（Dolphin.learn）在惊讶度算出后、入缓冲前，按相对当前能力基线
（ability）的偏离 delta 做三态判决。

带通定义（律定域内值自成）：
    delta = (surprise - ability) / ability    # 相对偏离，无量纲
    delta <  -GOLDILOCKS_LO  → 太熟（已会）   → 跳过，不进缓冲
    delta >  +GOLDILOCKS_HI  → 太怪（学不会） → 挂起，进记忆库冷层
    其余                      → 中间带         → 放行进缓冲

带是「以能力基线为中心、略偏难」的区间 [ability×(1−LO), ability×(1+HI)]：
数据惊喜度至少高于基线 5% 才值得学（太低=已会），高于 20% 即挂起（太怪=还
学不会，进冷层留待能力增长后再碰）。冷启动安全：随机初始化模型的惊讶度
≈ ln256 ≈ ability 初始值，delta≈0 落在中间带内，数据正常放行。
"""
import math

# 律定域内值自成（初值）：太熟/太怪的相对偏离阈值。
GOLDILOCKS_LO = 0.05   # 惊喜度比基线低 5% = 太熟（已会）→ 跳过
GOLDILOCKS_HI = 0.20   # 惊喜度比基线高 20% = 太怪（学不会）→ 挂起

# ability 初始值 / 旧档缺省值：随机初始化模型对任意字节流的理论惊讶度。
PRIOR_ABILITY = math.log(256)  # ≈5.545


def classify_delta(delta):
    """按相对偏离 delta 判金发姑娘三态。

    返回 "skip"（太熟跳过）/ "suspend"（太怪挂起）/ "pass"（中间带放行）。
    """
    if delta < -GOLDILOCKS_LO:
        return "skip"
    if delta > GOLDILOCKS_HI:
        return "suspend"
    return "pass"


def gate(surprise, ability):
    """金发姑娘硬门控：返回 (状态, delta)。状态 ∈ {"skip","suspend","pass"}。

    ability 是当前能力基线（非负；随睡眠周期慢速 EMA 更新）。surprise 是
    单条经验经当前醒脑算出的惊讶度（mean_nll）。
    """
    delta = (surprise - ability) / ability
    return classify_delta(delta), delta
