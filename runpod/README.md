# RunPod training

For GitHub Actions provisioning, see [Actions setup](actions.md). That workflow
uses the repository Secret `RUNPOD_API_KEY` and rents a GPU only on explicit manual
dispatch. The commands below are for an already configured GPU Pod. Real RunPod
CUDA validation is still pending; offline tests do not establish GPU compatibility.

## 1. Environment and persistent storage

Use Linux, Python 3.10–3.12, and working CUDA. Keep the image's CUDA-enabled torch
installation if it satisfies `torch>=2.6,<3`; the tested CPU baseline is 2.6.0.
Record the exact image and package versions. Use your persistent volume for data,
manifests, and checkpoints. `/workspace` below is an example mount point: verify
it against your Pod setup.

```bash
cd /workspace
git clone https://github.com/jasonjiang8866/spleeterpt.git
cd spleeterpt
git switch main
python -m pip install -e '.[dev]'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python scripts/smoke_test.py --device cuda --precision fp32
```

After fp32 succeeds, try `--precision bf16` on a supported GPU, or `fp16`. The smoke
model is deliberately tiny and does not estimate full-model VRAM needs. Do not
replace the CUDA build with a CPU wheel on the Pod.

## 2. Data

Copy/extract aligned stereo 44.1 kHz WAV stems onto the volume. Follow the root
README. Split only the training pool, preserving the independent test set:

```bash
spleeterpt manifest --root /workspace/data/train-pool \
  --output /workspace/data/manifests --validation-tracks 10 --seed 0
```

`10` is an example, not a benchmark-specific split. Supply standard benchmark
track-level manifests if needed. Five-stem training needs separate piano ground
truth; standard MUSDB is four stems.

## 3. Full-model smoke run, then baseline

```bash
export TRAIN_MANIFEST=/workspace/data/manifests/train.jsonl
export VAL_MANIFEST=/workspace/data/manifests/validation.jsonl
export RUN_DIR=/workspace/runs/2stems-smoke
export CONFIG=configs/2stems.json
export BATCH_SIZE=1
export PRECISION=fp32
bash runpod/train.sh --max-steps 2 --save-every 1
```

Inspect finite loss, peak VRAM, checkpoint writing, and reloading. A two-step run
can exit before validation; its output is `last.pt`. For a full run choose a new
`RUN_DIR` and omit `--max-steps`. Add `--init /workspace/weights/2stems.pt` for
fine-tuning, or omit it for random initialization. Increase batch size only after
observing memory use on the chosen GPU.

```bash
export RUN_DIR=/workspace/runs/2stems-baseline
mkdir -p "$RUN_DIR"
python -m pip freeze > /workspace/runs/2stems-baseline-packages.txt
bash runpod/train.sh
```

Use a persistent terminal session (for example, tmux) to avoid losing the process
when disconnecting. Progress is also written to `$RUN_DIR/metrics.jsonl`.

## 4. Resume

Keep config, seed, batch size, precision, crop counts, data paths, and files unchanged:

```bash
bash runpod/train.sh --resume "$RUN_DIR/last.pt"
```

Checkpoints use atomic replacement. A process killed between saves loses only work
since the last completed checkpoint; partial epochs can resume. Changing paths,
mtime, or settings requires `--init` in a fresh run directory. CPU resume is tested
for identical weights; GPU bitwise equality is not promised across environments.

For a deterministic CUDA experiment set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and add
`--deterministic` on the original run and resume. PyTorch may reject operations
without deterministic kernels; measure throughput before a long run.

## 5. Before scaling up

Measure held-out quality and listening samples, record steps/sec and peak VRAM,
and test a real Pod restart/resume. This version uses single-device Adam;
multi-GPU/DDP, alternative optimizers, and SDR evaluation are follow-up work.
No cloud credentials belong in this repository.
