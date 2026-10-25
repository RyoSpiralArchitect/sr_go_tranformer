# Copyright 2026 RyoSpiralArchitect
# SPDX-License-Identifier: Apache-2.0
"""Check reference admission without installing Torch or allocating model weights."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock, patch


class ReferenceAdmission(unittest.TestCase):
    def test_failed_fp32_preflight_never_constructs_reference_model(self):
        torch = types.ModuleType('torch')
        torch.__version__ = '2.8.0'
        torch.set_num_threads = Mock()
        transformers = types.ModuleType('transformers')
        transformers.__version__ = '4.57.1'
        transformers.AutoTokenizer = Mock()
        transformers.AutoModelForCausalLM = Mock()
        transformers.LlamaForCausalLM = Mock()
        source = dict(id='oversized', repo='fixture/model', revision='fixture',
                      files={}, weight_mib=8)
        script = Path(__file__).with_name('hf_model_reference.py')
        spec = importlib.util.spec_from_file_location('model_reference_admission', script)
        ref = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, torch=torch, transformers=transformers,
                        numpy=types.ModuleType('numpy')):
            spec.loader.exec_module(ref)
            with patch.object(sys, 'argv', [str(script), '--model', 'unused-model',
                                           '--id', 'oversized', '--binary', 'unused-binary',
                                           '--output', 'unused-output']), \
                    patch.object(Path, 'read_text', return_value=json.dumps({'models': [source]})), \
                    patch.object(ref.subprocess, 'check_output',
                                 side_effect=subprocess.CalledProcessError(1, 'preflight')) as check:
                with self.assertRaises(subprocess.CalledProcessError):
                    ref.main()
        command = check.call_args.args[0]
        self.assertEqual(command[1], 'inspect-hf')
        self.assertEqual(command[command.index('-weight-mib')+1], '8')
        self.assertNotIn('-threads', command)
        transformers.AutoTokenizer.from_pretrained.assert_not_called()
        transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
        torch.set_num_threads.assert_not_called()


if __name__ == '__main__':
    unittest.main()
