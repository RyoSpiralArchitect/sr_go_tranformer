import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quant_quality import git_source, source_path


def documentation_files(root):
    return sorted(root.glob('*.md')) + [
        document
        for folder in ('docs', 'tools', 'examples', 'testdata')
        for document in sorted((root / folder).rglob('*.md'))
    ]


class RepositoryLayoutTests(unittest.TestCase):
    def test_current_sources_resolve(self):
        root = Path(__file__).resolve().parents[1]
        sources = sorted((root / 'cmd/monolith').glob('*.go'))
        self.assertTrue(sources)
        for source in sources:
            with self.subTest(source=source.name):
                self.assertEqual(source_path(root, source.name), source)
        self.assertEqual(source_path(root, 'go.mod'), root / 'go.mod')
        self.assertEqual(source_path(root, 'scripts/go.sh'), root / 'scripts/go.sh')

    def test_legacy_checkout_sources_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'monolith.go'
            source.touch()
            self.assertEqual(source_path(root, source.name), source)

    def test_git_source_supports_both_layouts(self):
        root = Path(__file__).resolve().parents[1]
        for moved in (False, True):
            with self.subTest(moved=moved), patch('quant_quality.subprocess.run') as run:
                success = subprocess.CompletedProcess([], 0, b'package main', b'')
                failure = subprocess.CompletedProcess([], 128, b'', b'not found')
                run.side_effect = [failure, success] if moved else [success]
                self.assertEqual(git_source(root, 'revision', 'monolith.go'), b'package main')
                self.assertEqual(run.call_count, 2 if moved else 1)
                expected = 'revision:cmd/monolith/monolith.go' if moved else 'revision:monolith.go'
                self.assertEqual(run.call_args.args[0][-1], expected)

    def test_missing_git_source_fails(self):
        with patch('quant_quality.subprocess.run') as run:
            run.return_value = subprocess.CompletedProcess(['git', 'show'], 128, b'', b'not found')
            with self.assertRaises(subprocess.CalledProcessError):
                git_source(Path('.'), 'revision', 'monolith.go')

    def test_documentation_links_resolve(self):
        root = Path(__file__).resolve().parents[1]
        for document in documentation_files(root):
            for target in re.findall(r'\]\(([^)]+)\)', document.read_text()):
                if '://' in target or target.startswith('#'):
                    continue
                path = target.split('#', 1)[0]
                with self.subTest(document=str(document.relative_to(root)), target=target):
                    self.assertTrue((document.parent / path).exists())

    def test_documented_builds_use_command_package(self):
        root = Path(__file__).resolve().parents[1]
        pattern = r'(?:go|go\.sh)\s+(?:build[^\n`]*-o\s+monolith|run)\s+\.(?=[`\s]|$)'
        for document in (root / 'docs').glob('*.md'):
            with self.subTest(document=document.name):
                self.assertIsNone(re.search(pattern, document.read_text()))


if __name__ == '__main__':
    unittest.main()
