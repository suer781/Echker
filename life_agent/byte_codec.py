"""Self-describing byte frames and a single universal next-byte objective.

The model consumes bytes, never Python strings, tensors of pixels, or audio
objects. Codec classes below only construct/parse containers; they do not
introduce modality-specific networks or losses.
"""
from __future__ import annotations

import base64
import codecs
from dataclasses import dataclass
from enum import IntEnum
import json
import mimetypes
from pathlib import Path
import struct
from typing import Any, Iterable, Iterator
import zlib

BOS = 256
EOS = 257
USER = 258
ASSISTANT = 259
SEP = 260
PAD = 261
VOCAB_SIZE = 262
CONTROL_IDS = {BOS, EOS, USER, ASSISTANT, SEP, PAD}
IGNORE_INDEX = -100

MAGIC = b"BLF1"
HEADER = struct.Struct(">4sBBHQI")  # magic, codec, flags, metadata bytes, payload bytes, CRC32


class Codec(IntEnum):
    RAW = 0
    UTF8 = 1
    RGB = 2
    PCM = 3


@dataclass(frozen=True)
class ByteFrame:
    """A codec byte, JSON metadata bytes, arbitrary payload, and CRC32.

    Custom codec IDs 4..255 are permitted. The learning core does not need to
    understand them: metadata and payload are still just bytes.
    """

    codec: int
    metadata: dict[str, Any]
    payload: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "codec", int(self.codec))
        object.__setattr__(self, "payload", bytes(self.payload))
        if not 0 <= self.codec <= 255:
            raise ValueError("codec must fit in one byte")
        if not isinstance(self.metadata, dict):
            raise TypeError("frame metadata must be a JSON object")

    @classmethod
    def text(cls, value: str) -> "ByteFrame":
        return cls(Codec.UTF8, {"encoding": "utf-8"}, value.encode("utf-8", errors="strict"))

    @classmethod
    def raw(cls, payload: bytes, metadata: dict[str, Any] | None = None) -> "ByteFrame":
        return cls(Codec.RAW, metadata or {"media_type": "application/octet-stream"}, payload)

    @classmethod
    def rgb(cls, payload: bytes, width: int, height: int, channels: int = 3) -> "ByteFrame":
        if width < 1 or height < 1 or channels < 1:
            raise ValueError("RGB dimensions and channels must be positive")
        expected = width * height * channels
        if len(payload) != expected:
            raise ValueError(f"RGB payload has {len(payload)} bytes; expected {expected}")
        return cls(
            Codec.RGB,
            {
                "width": width,
                "height": height,
                "channels": channels,
                "layout": "HWC",
                "dtype": "uint8",
            },
            payload,
        )

    @classmethod
    def pcm(
        cls,
        payload: bytes,
        sample_rate: int,
        sample_width: int,
        channels: int,
        byte_order: str = "little",
    ) -> "ByteFrame":
        if sample_rate < 1 or sample_width < 1 or channels < 1:
            raise ValueError("PCM format values must be positive")
        if len(payload) % (sample_width * channels):
            raise ValueError("PCM payload ends with an incomplete sample frame")
        return cls(
            Codec.PCM,
            {
                "sample_rate": sample_rate,
                "sample_width": sample_width,
                "channels": channels,
                "sample_frames": len(payload) // (sample_width * channels),
                "byte_order": byte_order,
            },
            payload,
        )

    @classmethod
    def from_file(cls, path: Path) -> "ByteFrame":
        if not path.is_file():
            raise ValueError(f"not a regular file: {path}")
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return cls.raw(path.read_bytes(), {"name": path.name, "media_type": media_type})

    def _metadata_bytes(self) -> bytes:
        encoded = json.dumps(
            self.metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(encoded) > 65535:
            raise ValueError("frame metadata exceeds 65535 bytes")
        return encoded

    def to_bytes(self) -> bytes:
        metadata = self._metadata_bytes()
        checksum = zlib.crc32(metadata)
        checksum = zlib.crc32(self.payload, checksum) & 0xFFFFFFFF
        header = HEADER.pack(MAGIC, self.codec, 0, len(metadata), len(self.payload), checksum)
        return header + metadata + self.payload

    def descriptor(self) -> bytes:
        """Output specification supplied before generation; length is zero."""
        return ByteFrame(self.codec, self.metadata, b"").to_bytes()

    def empty(self) -> "ByteFrame":
        return ByteFrame(self.codec, self.metadata, b"")

    def with_payload(self, payload: bytes) -> "ByteFrame":
        return ByteFrame(self.codec, self.metadata, payload)

    def expected_payload_length(self) -> int | None:
        """Read structural output length from metadata without a model-specific loss."""
        if "payload_length" in self.metadata:
            return int(self.metadata["payload_length"])
        if self.codec == Codec.RGB:
            return int(self.metadata["width"]) * int(self.metadata["height"]) * int(self.metadata["channels"])
        if self.codec == Codec.PCM and "sample_frames" in self.metadata:
            return int(self.metadata["sample_frames"]) * int(self.metadata["sample_width"]) * int(self.metadata["channels"])
        return None

    def validate_payload(self) -> None:
        expected = self.expected_payload_length()
        if expected is not None and len(self.payload) != expected:
            raise ValueError(f"payload has {len(self.payload)} bytes; frame structure requires {expected}")

    def as_text(self, errors: str = "strict") -> str:
        if self.codec != Codec.UTF8:
            raise ValueError("frame is not declared as UTF-8 text")
        return self.payload.decode("utf-8", errors=errors)

    def to_dict(self) -> dict[str, str]:
        return {"frame": base64.b64encode(self.to_bytes()).decode("ascii")}

    @classmethod
    def from_dict(cls, raw: dict[str, str]) -> "ByteFrame":
        return cls.from_bytes(base64.b64decode(raw["frame"]))

    @classmethod
    def from_bytes(cls, data: bytes) -> "ByteFrame":
        if len(data) < HEADER.size:
            raise ValueError("truncated byte-frame header")
        magic, codec, flags, metadata_length, payload_length, expected_crc = HEADER.unpack_from(data)
        if magic != MAGIC:
            raise ValueError("invalid byte-frame magic")
        if flags != 0:
            raise ValueError("unsupported byte-frame flags")
        expected_size = HEADER.size + metadata_length + payload_length
        if len(data) != expected_size:
            raise ValueError(f"byte-frame length mismatch: expected {expected_size}, got {len(data)}")
        metadata_start = HEADER.size
        payload_start = metadata_start + metadata_length
        metadata_bytes = data[metadata_start:payload_start]
        payload = data[payload_start:]
        checksum = zlib.crc32(metadata_bytes)
        checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
        if checksum != expected_crc:
            raise ValueError("byte-frame CRC32 mismatch")
        metadata = json.loads(metadata_bytes.decode("utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("byte-frame metadata must be a JSON object")
        return cls(codec, metadata, payload)


def encode_text(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="strict"))


def decode_bytes(values: Iterable[int]) -> str:
    raw = bytes(int(value) for value in values if 0 <= int(value) < 256)
    return raw.decode("utf-8", errors="replace")


def build_frame_prompt(
    history: list[tuple[ByteFrame, ByteFrame]],
    stimulus: ByteFrame,
    response_spec: ByteFrame,
    max_seq_len: int,
    context_turns: int,
) -> list[int]:
    ids = [BOS]
    for past_stimulus, past_response in history[-context_turns:]:
        ids.extend((USER, *past_stimulus.to_bytes(), SEP, ASSISTANT))
        ids.extend((*past_response.descriptor(), *past_response.payload, EOS))
    ids.extend((USER, *stimulus.to_bytes(), SEP, ASSISTANT, *response_spec.descriptor()))
    if len(ids) > max_seq_len:
        # Preserve the current stimulus role boundary while sliding the byte
        # window. Frame headers may slide out, but the dialogue structure may not.
        ids = [BOS, USER, *ids[-(max_seq_len - 2):]]
    return ids


def frame_dialogue_sample(
    stimulus: ByteFrame, response: ByteFrame, response_spec: ByteFrame | None = None
) -> tuple[list[int], list[int]]:
    """The universal loss predicts response payload bytes, regardless of codec."""
    specification = response_spec or response.empty()
    prefix = [
        BOS,
        USER,
        *stimulus.to_bytes(),
        SEP,
        ASSISTANT,
        *specification.descriptor(),
    ]
    target = [*response.payload, EOS]
    return prefix + target, [IGNORE_INDEX] * len(prefix) + target


def observation_sample(frame: ByteFrame) -> tuple[list[int], list[int]]:
    ids = [BOS, USER, *frame.to_bytes(), EOS]
    return ids, ids.copy()


def observation_chunks(frame: ByteFrame, max_seq_len: int) -> Iterator[tuple[list[int], list[int]]]:
    """Stream an entire frame without changing the objective or truncating data."""
    payload = list(frame.to_bytes())
    width = max_seq_len - 3
    if width < 1:
        raise ValueError("context is too short for observation framing")
    for start in range(0, len(payload), width):
        ids = [BOS, USER, *payload[start:start + width], EOS]
        yield ids, ids.copy()


def build_prompt(
    history: list[tuple[str, str]], user: str, max_seq_len: int, context_turns: int
) -> list[int]:
    """Compatibility wrapper implemented entirely through universal frames."""
    frame_history = [(ByteFrame.text(stimulus), ByteFrame.text(response)) for stimulus, response in history]
    return build_frame_prompt(
        frame_history, ByteFrame.text(user), ByteFrame.text(""), max_seq_len, context_turns
    )


def dialogue_sample(user: str, assistant: str | Iterable[int]) -> tuple[list[int], list[int]]:
    assistant_bytes = assistant.encode("utf-8") if isinstance(assistant, str) else bytes(
        int(value) for value in assistant if 0 <= int(value) < 256
    )
    return frame_dialogue_sample(
        ByteFrame.text(user), ByteFrame(Codec.UTF8, {"encoding": "utf-8"}, assistant_bytes)
    )


def user_sample(user: str) -> tuple[list[int], list[int]]:
    return observation_sample(ByteFrame.text(user))


def trim_training_pair(
    ids: list[int], labels: list[int], max_seq_len: int
) -> tuple[list[int], list[int]]:
    if len(ids) < 2:
        raise ValueError("a training sequence needs at least two tokens")
    ids = ids[-max_seq_len:]
    labels = labels[-max_seq_len:]
    if all(label == IGNORE_INDEX for label in labels[1:]):
        labels = ids.copy()
    return ids, labels


class IncrementalUTF8Decoder:
    """Prevents replacement characters at display chunk boundaries."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token: int, final: bool = False) -> str:
        if 0 <= token < 256:
            return self._decoder.decode(bytes((token,)), final=final)
        return self._decoder.decode(b"", final=final)
