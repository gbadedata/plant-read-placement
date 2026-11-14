# Runbook

The fastest route is to run the CPU stages (downloads, simulation, minimap2) on any machine with
about 32 GB of RAM and 60 GB of free disk, then use a GPU machine only for PlantCaduceus. Everything
can also run on one GPU VM.

## 0. Machines

**CPU stages:** your laptop (Ubuntu under WSL is fine) or any Linux VM with at least 4 cores and 32 GB RAM.

**GPU stages:** one NVIDIA GPU, Ampere or newer (A10, L4, A100). PlantCaduceus's authors report speeds
only on Ampere-class cards; a T4 may work with `amp: fp16` but is untested. On Azure:

1. Portal → **Quotas → Compute** → your region → request vCPUs for an A10 or A100 family (for example
   "Standard NVADSA10v5 Family" or "Standard NCADS_A100_v4 Family"). Approval can take days.
2. Create the VM from an Ubuntu 22.04 image with the NVIDIA GPU driver extension (or the NVIDIA
   GPU-optimised image), 128 GB OS disk. Check with `nvidia-smi`.
3. **Stop (deallocate) the VM whenever it is idle.** You pay for a running GPU VM by the hour.

## 1. Install

```bash
# conda (Miniforge) if you do not have it
curl -L -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash mf.sh -b && source ~/miniforge3/bin/activate

git clone https://github.com/gbadedata/plant-read-placement && cd plant-read-placement
bash scripts/setup_env.sh          # CPU tools: package, Badread, NCBI Datasets CLI, minimap2 bindings
bash scripts/setup_env.sh --gpu    # on the GPU machine: adds transformers, mamba-ssm, causal-conv1d
conda activate prp
pytest                              # about a minute; includes the full toy pipeline
```

The GPU install follows PlantCAD's own instructions (`mamba-ssm==2.2.4 causal-conv1d==1.5.0.post8
--no-build-isolation`). If `from mamba_ssm import Mamba` fails, reinstall mamba-ssm alone without
causal-conv1d, as their troubleshooting section advises.

## 2. One-minute toy check (any machine)

```bash
prp toy --out toy && cd toy && prp run-toy --panel configs/panel.yaml --config configs/experiment.yaml && cd ..
```

This exercises every stage including freeze and the locked test run. Toy numbers mean nothing.

## 3. The real run

Run long steps inside `tmux` so a dropped connection does not kill them.

| # | command | where | rough time |
|---|---|---|---|
| 1 | `prp check-panel --strict` | CPU | seconds |
| 2 | `prp resolve` then read `work/genomes/lock.json` | CPU | minutes |
| 3 | `prp fetch` | CPU | 30-60 min (about 5 GB) |
| 4 | `prp split` | CPU | 10-20 min |
| 5 | `prp simulate --workers 4` | CPU | 1-3 h (Badread about 80 kb/s per worker) |
| 6 | `prp windows` | CPU | 10 min |
| 7 | `prp embed --encoder kmer` | CPU | 10-20 min |
| 8 | `prp minimap --splits val test` | CPU | 30-60 min |
| 9 | copy `configs/` and `work/windows/` to the GPU machine | | |
| 10 | `prp embed --encoder plantcad` | GPU | 30-90 min |
| 11 | `prp list-params --encoder plantcad`, then set `finetune.trainable_regex` | GPU | minutes |
| 12 | `prp finetune --name plantcad_adapted` | GPU | 30-90 min |
| 13 | `prp embed --encoder plantcad --checkpoint work/models/plantcad_adapted/encoder.pt --name plantcad_ft` | GPU | 30-90 min |
| 14 | copy `work/features/` and `work/models/` back | | |
| 15 | `prp train-head --features kmer`, `... plantcad`, `... plantcad_ft` | GPU or CPU | 5-15 min each |
| 16 | `prp dev --methods kmer_raw kmer_head plantcad_raw plantcad_head plantcad_ft_head minimap2` | CPU | 10 min |
| 17 | fill every `TBD` in `PREREG.md` from `work/results/val/RESULTS.md`; commit and push | | |
| 18 | `prp freeze`; commit `prereg.lock.json`; push | CPU | seconds |
| 19 | `prp evaluate --split test --methods <same list>` | CPU | 10 min |
| 20 | copy `work/results/test/` to `results/`; commit `TEST_RUN.json` and results; push | | |

Pushing after steps 17 and 18 gives a public, timestamped record that the targets came before the
test run.

### Notes per step

- **2 resolve:** if a species prints `NO ASSEMBLY FOUND`, use the alternate given in its `note` in
  `configs/panel.yaml`, or pin an accession with `accession: GCF_...`. Re-run `prp check-panel --strict`.
- **5 simulate:** each Badread worker holds one split's region FASTA in memory (up to about 1 GB of
  sequence); four workers is safe with 32 GB. `--backend builtin` is much faster but less realistic;
  if you use it, the write-up must say so.
- **10 embed:** if you see out-of-memory errors, lower `encoders.plantcad.batch_size`. On a GPU
  without bf16 (T4, V100) set `amp: fp16`.
- **11 list-params:** pick the last four backbone blocks and the final norm. The default patterns
  are a guess at Caduceus's names; the run stops with a list of names if nothing matches.
- **16 dev:** this is the only place you look at numbers before freezing. Change anything you want
  here (hyperparameters, steps, which methods), then re-run dev. Once you freeze, you stop.
- **19 evaluate:** a second test run is refused unless you pass `--rerun-reason "..."`, which is
  recorded as a deviation. Use it only for genuine bugs and say so in the write-up.

## 4. If time is short

Minimum credible run, in priority order:
1. `plantcad_head`, `kmer_head`, `minimap2` with `specimen_sizes: [1, 10]`.
2. Add `plantcad_ft_head` (the adaptation result).
3. Add `plantcad_raw`, `kmer_raw` and n = 50.

Reduce `simulate.reads` to `{train: 2000, val: 600, test: 600}` to halve simulation and embedding
time. Make these changes before freezing.

## 5. Troubleshooting

| symptom | fix |
|---|---|
| `datasets: command not found` | `conda install -c conda-forge ncbi-datasets-cli` |
| `badread not on PATH` | `pip install git+https://github.com/rrwick/Badread.git` |
| `cannot locate nucleotide tokens` | the tokenizer is not character-level; check the model id |
| `trainable_regex matched no parameters` | run `prp list-params` and copy the real names |
| `files changed since the preregistration was frozen` | you edited a frozen file; restore it from git |
| PlantCaduceus very slow | confirm `torch.cuda.is_available()`; CPU inference is supported but much slower |
