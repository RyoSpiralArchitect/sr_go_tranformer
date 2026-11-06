import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from compare_inference import isolated_go_environment
from quant_quality import git_source, sha


PROBE = '''package PACKAGE

import "context"
import "strings"

func NewGenerationProbe() (func(int, int) (any, error), error) {
    c := Config{Vocab: 258, Dim: 256, Hidden: 704, Layers: 1, Heads: 4, KVHeads: 2, Context: 512, RopeTheta: 10000, NormEps: 1e-5}
    m, err := NewModel(c, 19)
    if err != nil { return nil, err }
    m.DropOptimizer()
    return func(prefix, outputs int) (any, error) {
        s := defaultSampling()
        s.MaxTokens, s.Temperature, s.RepeatPenalty, s.Seed = outputs, 0, 1, 19
        return generateInput(context.Background(), m, fixedStepGenerateTokenizer{}, strings.Repeat("a", prefix-1), s, false)
    }, nil
}
'''

DRIVER = '''package main

import (
    "bytes"
    "encoding/json"
    "flag"
    "fmt"
    "os"
    "runtime"
    "sort"
    "time"
    before "MODULE/before"
    after "MODULE/after"
)

type result struct {
    Prefix int `json:"prefix"`
    Outputs int `json:"outputs"`
    BeforeNS []float64 `json:"before_ns"`
    AfterNS []float64 `json:"after_ns"`
    Ratios []float64 `json:"ratios"`
    MedianRatio float64 `json:"median_ratio"`
}

func check(a, b any) {
    x, err := json.Marshal(a)
    if err != nil { panic(err) }
    y, err := json.Marshal(b)
    if err != nil { panic(err) }
    if !bytes.Equal(x, y) { panic("completion changed") }
}

func main() {
    output := flag.String("output", "", "fresh JSON receipt path")
    trials := flag.Int("trials", 24, "paired trials")
    flag.Parse()
    if *output == "" || *trials < 2 || *trials > 128 { panic("output and 2..128 trials required") }
    runtime.GOMAXPROCS(4)
    file, err := os.OpenFile(*output, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
    if err != nil { panic(err) }
    defer file.Close()
    first, err := before.NewGenerationProbe()
    if err != nil { panic(err) }
    second, err := after.NewGenerationProbe()
    if err != nil { panic(err) }
    functions := []func(int, int) (any, error){first, second}
    var results []result
    for _, shape := range [][2]int{{3, 1}, {129, 1}, {257, 1}, {33, 8}, {33, 128}} {
        prefix, outputs := shape[0], shape[1]
        for warmup := 0; warmup < 3; warmup++ {
            a, err := first(prefix, outputs)
            if err != nil { panic(err) }
            b, err := second(prefix, outputs)
            if err != nil { panic(err) }
            check(a, b)
        }
        runtime.GC()
        row := result{Prefix: prefix, Outputs: outputs}
        for trial := 0; trial < *trials; trial++ {
            var elapsed [2]float64
            var completions [2]any
            for position := 0; position < 2; position++ {
                variant := (trial + position) % 2
                start := time.Now()
                value, err := functions[variant](prefix, outputs)
                elapsed[variant] = float64(time.Since(start).Nanoseconds())
                if err != nil { panic(err) }
                completions[variant] = value
            }
            check(completions[0], completions[1])
            row.BeforeNS = append(row.BeforeNS, elapsed[0])
            row.AfterNS = append(row.AfterNS, elapsed[1])
            row.Ratios = append(row.Ratios, elapsed[1]/elapsed[0])
        }
        ratios := append([]float64(nil), row.Ratios...)
        sort.Float64s(ratios)
        row.MedianRatio = (ratios[(len(ratios)-1)/2] + ratios[len(ratios)/2])/2
        results = append(results, row)
        fmt.Printf("prefix%d/outputs%d: paired ratio %.3f\\n", prefix, outputs, row.MedianRatio)
    }
    encoder := json.NewEncoder(file)
    encoder.SetIndent("", "  ")
    if err := encoder.Encode(results); err != nil { panic(err) }
    if err := file.Sync(); err != nil { panic(err) }
}
'''


def namespace_source(content, package):
    if package not in ('before', 'after') or content.count(b'package main\n') != 1:
        raise ValueError('unexpected package declaration')
    return content.replace(b'package main\n', ('package ' + package + '\n').encode(), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', choices=('native', 'linux-amd64'), default='native')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = isolated_go_environment()
    baseline = subprocess.check_output(['git', 'rev-parse', '--verify', '--end-of-options',
                                         args.baseline_ref + '^{commit}'], cwd=root, text=True).strip()
    goroot = subprocess.check_output([root / 'scripts/go.sh', 'env', 'GOROOT'],
                                     cwd=root, env=env, text=True).strip()
    go = str(Path(goroot) / 'bin/go')
    for name in ('go.mod', 'go.sum'):
        (out / name).write_bytes((root / name).read_bytes())
    module = re.search(r'^module\s+(\S+)', (out / 'go.mod').read_text(), re.MULTILINE).group(1)
    driver_hash = sha(Path(__file__))
    candidate = (root / 'cmd/monolith/monolith.go').read_bytes()
    previous = git_source(root, baseline, 'monolith.go')
    shared = {'test_' + path.name.removesuffix('_test.go') + '.go': path.read_bytes()
              for path in sorted((root / 'cmd/monolith').glob('*_test.go'))
              if path.name != 'executor_experiments_test.go'}
    for variant in ('before', 'after'):
        build = out / variant
        build.mkdir()
        sources = {'monolith.go': previous if variant == 'before' else candidate, **shared}
        for name, content in sources.items():
            (build / name).write_bytes(namespace_source(content, variant))
        (build / 'generation_probe.go').write_text(PROBE.replace('PACKAGE', variant))
    (out / 'main.go').write_text(DRIVER.replace('MODULE', module))
    if args.target == 'linux-amd64':
        env.update(GOOS='linux', GOARCH='amd64')
    inputs = {str(p.relative_to(out)): sha(p) for p in [*sorted(out.rglob('*.go')), out / 'go.mod', out / 'go.sum']}
    subprocess.run([go, 'build', '-trimpath', '-o', out / 'generation-pair', '.'],
                   cwd=out, env=env, check=True, timeout=300)
    observed = {str(p.relative_to(out)): sha(p) for p in [*sorted(out.rglob('*.go')), out / 'go.mod', out / 'go.sum']}
    if inputs != observed or driver_hash != sha(Path(__file__)):
        raise ValueError('comparison sources changed during compilation')
    receipt = dict(format='monolith-dual-generation-build-v1', baseline=baseline,
                   candidate_runtime_sha256=hashlib.sha256(candidate).hexdigest(),
                   go_version=subprocess.check_output([go, 'version'], text=True).strip(),
                   target=args.target, driver_sha256=driver_hash,
                   binary_sha256=sha(out / 'generation-pair'), inputs_sha256=inputs,
                   limitations='Two package-renamed runtimes in one executable, synthetic FP32 model. Completion adapters box results into interfaces inside timing; JSON equality checks run outside timing. No allocation/RSS, real-model or serving-speed claim.')
    encoded = json.dumps(receipt, indent=2) + '\n'
    (out / 'build-receipt.json').write_text(encoded)
    if args.receipt:
        with args.receipt.open('x') as output:
            output.write(encoded)
    print(out / 'generation-pair', flush=True)


if __name__ == '__main__':
    main()
