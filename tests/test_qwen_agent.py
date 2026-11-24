import unittest

from qwen_agent import completion_labels


class CompletionLabelTests(unittest.TestCase):
    def test_only_assistant_completion_is_trainable(self):
        labels = completion_labels([10, 11], [10, 11, 12, 13])
        self.assertEqual(labels, [-100, -100, 12, 13])

    def test_rejects_non_prefix_chat_template(self):
        with self.assertRaisesRegex(ValueError, "not a prefix"):
            completion_labels([10, 11], [10, 12, 13])

    def test_rejects_empty_completion(self):
        with self.assertRaisesRegex(ValueError, "non-empty answer"):
            completion_labels([10, 11], [10, 11])


if __name__ == "__main__":
    unittest.main()
