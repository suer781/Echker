"""A local, dual-hemisphere, byte-level continual-learning research prototype."""

from .agent import LifeAgent
from .byte_codec import ByteFrame, Codec
from .config import AppConfig, ModelConfig, RuntimeConfig

__all__ = ["LifeAgent", "ByteFrame", "Codec", "AppConfig", "ModelConfig", "RuntimeConfig"]
__version__ = "0.5.0"
