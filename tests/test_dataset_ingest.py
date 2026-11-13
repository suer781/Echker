from pathlib import Path
import tempfile
import unittest

from ingest_dataset import DatasetStreamer, ProgressStore

PROJECT = Path(__file__).resolve().parents[1]


class DatasetStreamingTests(unittest.TestCase):
    def test_chunks_preserve_raw_bytes_and_resume_at_offset(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            content = bytes(range(256)) * 2
            (root / "a.bin").write_bytes(content)
            (root / "b.txt").write_bytes(b"second")
            state = Path(directory) / "state"
            streamer = DatasetStreamer(root, chunk_bytes=100, holdout_fraction=0, includes=[], excludes=[])
            progress = ProgressStore(state / "progress.json", root, fresh=True)
            inventory = streamer.inventory()
            self.assertEqual(inventory["total_files"], 2)
            self.assertEqual(inventory["total_bytes"], len(content) + 6)

            iterator = streamer.frames(progress, max_bytes=100, max_files=0, deadline=None)
            frame, record, offset = next(iterator)
            self.assertEqual(record.relative, "a.bin")
            self.assertEqual(frame.payload, content[:100])
            self.assertEqual(frame.metadata["offset"], 0)
            progress.mark(record, offset)
            iterator.close()

            resumed = ProgressStore(state / "progress.json", root)
            iterator = streamer.frames(resumed, max_bytes=100, max_files=0, deadline=None)
            frame, record, offset = next(iterator)
            self.assertEqual(frame.metadata["offset"], 100)
            self.assertEqual(frame.payload, content[100:200])
            self.assertEqual(offset, 200)
            iterator.close()

    def test_completed_files_are_not_retrained(self):
        with tempfile.TemporaryDirectory(dir=PROJECT / "tests") as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            (root / "small.bin").write_bytes(b"abc")
            progress = ProgressStore(Path(directory) / "progress.json", root, fresh=True)
            streamer = DatasetStreamer(root, 100, 0, [], [])
            for frame, record, offset in streamer.frames(progress, 0, 0, None):
                self.assertEqual(frame.payload, b"abc")
                progress.mark(record, offset)
            resumed = ProgressStore(Path(directory) / "progress.json", root)
            self.assertEqual(list(streamer.frames(resumed, 0, 0, None)), [])


if __name__ == "__main__":
    unittest.main()
