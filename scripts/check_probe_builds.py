import subprocess
import tempfile
from pathlib import Path

from compare_inference import isolated_go_environment
from quant_quality import source_path


def main():
    root = Path(__file__).resolve().parents[1]
    env = isolated_go_environment()
    goroot = subprocess.check_output(
        [root / 'scripts/go.sh', 'env', 'GOROOT'], cwd=root, env=env, text=True
    ).strip()
    go = str(Path(goroot) / 'bin/go')
    probes = {
        'inference': ['inference_perf_test.go'],
        'long-generation': ['inference_perf_test.go', 'long_generation_perf_test.go'],
        'prompt-prefill': ['prompt_prefill_perf_test.go'],
        'prefix-chat': ['prefix_chat_perf_test.go'],
        'prompt-kv': ['inference_perf_test.go', 'testdata/probes/prompt_kv_test.go'],
    }
    with tempfile.TemporaryDirectory(prefix='monolith-probes-') as directory:
        for name, harnesses in probes.items():
            build = Path(directory) / name
            build.mkdir()
            for source in ['monolith.go', 'go.mod', 'go.sum', *harnesses]:
                path = source_path(root, source)
                (build / path.name).write_bytes(path.read_bytes())
            binary = build / 'probe.test'
            subprocess.run([go, 'test', '-c', '-trimpath', '-o', binary, '.'],
                           cwd=build, env=env, check=True, timeout=300)
            subprocess.run([binary, '-test.run=^$'], cwd=root, env=env, check=True, timeout=30)
            print('PASS: isolated ' + name + ' probe build', flush=True)


if __name__ == '__main__':
    main()
