"""多模态输入适配（零重依赖）。
原理→代码映射：婴儿统计学习对声音流/视觉场景用同一机制（Saffran 1996；Fiser & Aslin 2002）——
多模态以原始字节进入同一条经验链路，金发姑娘统一去噪。"""


def encode_image_bytes(raw: bytes) -> bytes:
    """图片原始文件字节直通（模型靠暴露自学统计结构）。"""
    return raw


def encode_audio_bytes(raw: bytes) -> bytes:
    """音频原始文件字节直通。"""
    return raw


def sample_video_chunks(raw: bytes, chunk=4096, max_chunks=64) -> list:
    """视频"自适应抽帧"的零依赖落地：文件字节按块切分，返回分块列表；
    金发姑娘带通在经验链路中自然筛选信息量大的块（重复/静止块低分）。"""
    return [raw[i:i + chunk] for i in range(0, len(raw), chunk)][:max_chunks]