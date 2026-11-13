import unittest

from life_agent.byte_codec import (
    ASSISTANT,
    EOS,
    IGNORE_INDEX,
    ByteFrame,
    Codec,
    frame_dialogue_sample,
    observation_chunks,
)


class ByteFrameTests(unittest.TestCase):
    def test_arbitrary_binary_and_custom_codec_round_trip(self):
        payload = bytes(range(256))
        frame = ByteFrame(77, {"name": "未知格式", "values": [1, 2, 3]}, payload)
        restored = ByteFrame.from_bytes(frame.to_bytes())
        self.assertEqual(restored.codec, 77)
        self.assertEqual(restored.metadata["name"], "未知格式")
        self.assertEqual(restored.payload, payload)

    def test_crc_detects_payload_damage(self):
        data = bytearray(ByteFrame.text("完整数据").to_bytes())
        data[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC32"):
            ByteFrame.from_bytes(bytes(data))

    def test_truncation_is_rejected(self):
        data = ByteFrame.raw(b"abcdef").to_bytes()
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            ByteFrame.from_bytes(data[:-1])

    def test_rgb_is_raw_bytes_with_shape_metadata(self):
        frame = ByteFrame.rgb(bytes(12), width=2, height=2)
        self.assertEqual(frame.codec, Codec.RGB)
        self.assertEqual(frame.metadata["layout"], "HWC")
        self.assertEqual(frame.payload, bytes(12))
        with self.assertRaises(ValueError):
            ByteFrame.rgb(bytes(11), width=2, height=2)

    def test_pcm_frame_alignment(self):
        frame = ByteFrame.pcm(b"\x00\x00\x01\x00", 16000, 2, 1)
        self.assertEqual(frame.metadata["sample_rate"], 16000)
        with self.assertRaises(ValueError):
            ByteFrame.pcm(b"\x00", 16000, 2, 1)

    def test_structured_output_length_is_self_described(self):
        specification = ByteFrame.rgb(bytes(12), 2, 2).empty()
        self.assertEqual(specification.expected_payload_length(), 12)
        specification.with_payload(bytes(12)).validate_payload()
        with self.assertRaisesRegex(ValueError, "requires 12"):
            specification.with_payload(bytes(11)).validate_payload()
        pcm = ByteFrame.pcm(bytes(8), 16000, 2, 1).empty()
        self.assertEqual(pcm.expected_payload_length(), 8)

    def test_same_next_byte_labels_for_non_text_modalities(self):
        stimulus = ByteFrame.rgb(bytes(12), 2, 2)
        response = ByteFrame.raw(b"\x00\xff\x10")
        ids, labels = frame_dialogue_sample(stimulus, response)
        assistant = ids.index(ASSISTANT)
        self.assertTrue(all(value == IGNORE_INDEX for value in labels[:assistant]))
        self.assertEqual(ids[-4:], [0, 255, 16, EOS])
        self.assertEqual(labels[-4:], [0, 255, 16, EOS])

    def test_streaming_chunks_cover_every_frame_byte(self):
        frame = ByteFrame.raw(bytes(range(256)), {"test": "chunks"})
        encoded = frame.to_bytes()
        chunks = list(observation_chunks(frame, max_seq_len=32))
        reconstructed = []
        for ids, labels in chunks:
            self.assertEqual(ids, labels)
            reconstructed.extend(ids[2:-1])
        self.assertGreater(len(chunks), 1)
        self.assertEqual(bytes(reconstructed), encoded)


if __name__ == "__main__":
    unittest.main()
