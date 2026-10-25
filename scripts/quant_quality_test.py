# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

from quant_quality import run_measured


@unittest.skipUnless(hasattr(os, 'wait4') and platform.system() in ('Darwin', 'Linux'), 'requires wait4 RSS units')
class PeakRSS(unittest.TestCase):
    def test_each_child_has_its_own_peak(self):
        results = []
        for size in (96, 4):
            with tempfile.TemporaryFile(mode='w+') as log:
                code = ('import json,os,resource\n'
                        f'b=bytearray({size} << 20)\n'
                        'for i in range(0,len(b),4096): b[i]=1\n'
                        'print(json.dumps(dict(rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid())))\n')
                result = run_measured([sys.executable, '-S', '-c', code], timeout=10, stdout=log)
                log.seek(0)
                child = json.load(log)
                factor = 1024 if platform.system() == 'Linux' else 1
                self.assertGreaterEqual(result['peak_rss_bytes'], child['rss'] * factor)
                self.assertGreater(result['elapsed_seconds'], 0)
                with self.assertRaises(ChildProcessError):
                    os.waitpid(child['pid'], os.WNOHANG)
                results.append(result)
        self.assertGreater(results[0]['peak_rss_bytes'], results[1]['peak_rss_bytes'] + (48 << 20))

    def test_timeout_kills_and_reaps_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp)/'pid'
            code = f'import os,time; open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(60)'
            with self.assertRaises(subprocess.TimeoutExpired):
                run_measured([sys.executable, '-S', '-c', code], timeout=.5)
            pid = int(pidfile.read_text())
            with self.assertRaises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_failed_child_is_not_a_successful_measurement(self):
        with self.assertRaises(subprocess.CalledProcessError) as error:
            run_measured([sys.executable, '-S', '-c', 'raise SystemExit(7)'], timeout=10)
        self.assertEqual(error.exception.returncode, 7)

    def test_undrained_pipe_is_rejected(self):
        with self.assertRaises(ValueError):
            run_measured([sys.executable, '-S', '-c', 'print("x" * 1000000)'], timeout=1, stdout=subprocess.PIPE)


if __name__ == '__main__':
    unittest.main()
