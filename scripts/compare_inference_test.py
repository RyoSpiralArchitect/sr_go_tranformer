# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from compare_inference import isolated_go_environment


class BuildIsolationTests(unittest.TestCase):
    def test_saved_and_process_go_flags_are_disabled(self):
        go = shutil.which('go')
        if not go:
            fallback = Path(__file__).resolve().parents[1]/'.tools/go/bin/go'
            if fallback.is_file():
                go = str(fallback)
        if not go:
            self.skipTest('Go toolchain unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'go.env'
            config.write_text('GOFLAGS=-gcflags=all=-N\nGOTOOLCHAIN=auto\n')
            # Neither a saved -N nor a process-level -race may silently enter
            # a benchmark binary that claims the ordinary compiler settings.
            with patch.dict(os.environ, {'GOENV': str(config), 'GOFLAGS': '-race',
                                         'GOWORK': '/nonexistent/workspace', 'GOTOOLCHAIN': 'auto'}):
                env = isolated_go_environment()
            got = json.loads(subprocess.check_output(
                [go, 'env', '-json', 'GOENV', 'GOFLAGS', 'GOWORK', 'GOTOOLCHAIN', 'GOPROXY'],
                env=env, text=True, timeout=10))
            self.assertEqual(env['GOENV'], 'off')
            self.assertEqual(got, dict(GOENV='', GOFLAGS='', GOWORK='off',
                                       GOTOOLCHAIN='local', GOPROXY='off'))
            self.assertEqual(config.read_text(), 'GOFLAGS=-gcflags=all=-N\nGOTOOLCHAIN=auto\n')


if __name__ == '__main__':
    unittest.main()
