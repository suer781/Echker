import unittest

from life_agent.byte_codec import (
    ASSISTANT,
    BOS,
    EOS,
    IGNORE_INDEX,
    SEP,
    USER,
    IncrementalUTF8Decoder,
    build_prompt,
    decode_bytes,
    dialogue_sample,
    encode_text,
)


class ByteCodecTests(unittest.TestCase):
    def test_multilingual_round_trip_without_tokenizer(self):
        text = "你好，byte life。🙂 Ελληνικά"
        self.assertEqual(decode_bytes(encode_text(text)), text)

    def test_raw_incomplete_utf8_is_not_replaced_in_training_target(self):
        raw = encode_text("你")
        ids, labels = dialogue_sample("问", raw[:1])
        self.assertEqual(ids[-2], raw[0])
        self.assertEqual(ids[-1], EOS)
        self.assertEqual(labels[-2], raw[0])

    def test_dialogue_masks_user_tokens(self):
        ids, labels = dialogue_sample("用户", "回答")
        assistant_position = ids.index(ASSISTANT)
        self.assertTrue(all(value == IGNORE_INDEX for value in labels[:assistant_position + 1]))
        self.assertNotEqual(labels[-1], IGNORE_INDEX)

    def test_prompt_is_bounded(self):
        prompt = build_prompt([("甲" * 100, "乙" * 100)] * 3, "问题", 64, 6)
        self.assertLessEqual(len(prompt), 64)
        self.assertEqual(prompt[0], BOS)
        self.assertIn(USER, prompt)
        self.assertIn(SEP, prompt)

    def test_incremental_decoder_preserves_split_character(self):
        decoder = IncrementalUTF8Decoder()
        raw = encode_text("你")
        self.assertEqual(decoder.push(raw[0]), "")
        self.assertEqual(decoder.push(raw[1]), "")
        self.assertEqual(decoder.push(raw[2], final=True), "你")


if __name__ == "__main__":
    unittest.main()
