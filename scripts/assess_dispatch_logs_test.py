import tempfile
import unittest
from pathlib import Path

from assess_dispatch_logs import log_header, read_samples


class DispatchLogTests(unittest.TestCase):
    def test_pass_requires_complete_nonzero_samples(self):
        row = 'BenchmarkFixture-4 100 125.5 ns/op 96 B/op 1 allocs/op\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log.txt'
            path.write_text(row + 'PASS\n')
            self.assertEqual(read_samples(path)[0]['ns_per_op'], 125.5)
            for text in (row, 'PASS\n', row + 'FAIL\nPASS\n',
                         row + 'PASS\npanic: bad\n', row.replace('125.5', '0') + 'PASS\n'):
                with self.subTest(text=text), self.assertRaises(ValueError):
                    path.write_text(text)
                    read_samples(path)

    def test_headers_must_be_present_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log.txt'
            text = 'goos: linux\ngoarch: amd64\ncpu: fixture\n'
            path.write_text(text)
            self.assertEqual(log_header(path), dict(goos='linux', goarch='amd64', cpu='fixture'))
            for invalid in ('', text + 'cpu: duplicate\n', text.replace('goarch: amd64\n', '')):
                with self.subTest(text=invalid), self.assertRaises(ValueError):
                    path.write_text(invalid)
                    log_header(path)


if __name__ == '__main__':
    unittest.main()
