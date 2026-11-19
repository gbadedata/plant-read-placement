# Runbook

The CPU stages (downloads, read simulation, k-mers, minimap2) run on a laptop or any Linux machine.
Only the PlantCaduceus steps need a GPU, about 2 to 4 hours in total, and the default route for
those is a free Kaggle notebook (`kaggle/plantcad_gpu.ipynb`).

## 0. Machines

**CPU stages:** Ubuntu (WSL on Windows is fine) with 4+ cores, 16+ GB RAM visible to Linux and about
60 GB free disk. In WSL, work in the Linux home folder (`~/...`), not under `/mnt/c/...`, which is
much slower for large files.

**GPU stages, in order of preference:**
1. **Kaggle notebook (free).** GPU T4 x2, about 30 GPU hours a week, internet on (needs a verified
   phone number). The notebook builds PlantCaduceus's own pinned environment and stops at a
   pre-flight check before the long steps.
2. **Google Colab (free).** Same notebook; T4 availability is not guaranteed.
3. **Rented GPU or an Azure GPU VM.** Ampere or newer (A10, L4, A100); run the same commands as the
   notebook after `bash scripts/setup_env.sh --gpu`. Azure needs a GPU quota request first.

## 1. Install (CPU machine)

```bash
# conda inside Linux/WSL, if `conda --version` does not already work
curl -L -o ~/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash ~/mf.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && exec bash

cd ~ && git clone https://github.com/gbadedata/plant-read-placement && cd plant-read-placement
bash scripts/setup_env.sh          # package, Badread, NCBI Datasets CLI, minimap2 bindings
conda activate prp
pytest                              # about a minute; includes the full toy pipeline
```

## 2. One-minute toy check (optional)

```bash
prp toy --out toy && cd toy && prp run-toy --panel configs/panel.yaml --config configs/experiment.yaml && cd ..
```

## 3. The real run

Run long steps inside `tmux` (`tmux new -s prp`; reattach with `tmux attach -t prp`).

| # | command | where | rough time |
|---|---|---|---|
| 1 | `prp check-panel --strict` | laptop | seconds |
| 2 | `prp resolve`, then review `work/genomes/lock.json` | laptop | minutes |
| 3 | `prp fetch` | laptop | 30-60 min (about 5 GB) |
| 4 | `prp split` | laptop | 10-20 min |
| 5 | `prp simulate --workers 3` | laptop | 2-4 h (Badread is slow; run overnight) |
| 6 | `prp windows` | laptop | 10 min |
| 7 | `bash scripts/pack_for_kaggle.sh` and upload `prp_windows.zip` as a **private** Kaggle dataset | laptop / browser | 10 min |
| 8 | `prp embed --encoder kmer` and `prp minimap --splits val test` | laptop, while Kaggle runs | 1-2 h |
| 9 | Kaggle notebook cells 1-5: setup and `gpu-check` | Kaggle | 15 min |
| 10 | Kaggle notebook cells 6-10: embed, fine-tune, re-embed, heads, pack | Kaggle | 2-4 h |
| 11 | download `gpu_outputs.zip`; `cd work && python -m zipfile -e /mnt/c/Users/<you>/Downloads/gpu_outputs.zip .` | laptop | minutes |
| 12 | `prp train-head --features kmer` | laptop | 5-10 min |
| 12b | fine-tune retry on Kaggle (cells 1-5, then 7-10 with `RUN_ATTEMPT_1_STEPS = False`); download `gpu_outputs_r2.zip`; `cd work && python -m zipfile -e /mnt/c/Users/<you>/Downloads/gpu_outputs_r2.zip .` | Kaggle / laptop | about 4 h |
| 12b' | if the retry ran with attempt 1's notebook cells (names `plantcad_adapted`/`plantcad_ft`): `python scripts/import_retry.py <zip>` instead of unzipping; it renames the retry to `plantcad_adapted_r2`/`plantcad_ft2`, leaves attempt 1 untouched and checks the re-made frozen embeddings match | laptop | a minute |
| 12c | `prp minimap --reference train --splits val test` (training-blocks-only index) | laptop | a few hours |
| 13 | `prp dev --methods kmer_raw kmer_head plantcad_raw plantcad_head plantcad_ft_head plantcad_ft2_head minimap2 minimap2_train` | laptop | 10 min |
| 13b | optional, exploratory: `prp explore --methods kmer_head plantcad_head plantcad_ft2_head minimap2 minimap2_train` (where each class goes and why) | laptop | seconds |
| 14 | fill every `TBD` in `PREREG.md` from `work/results/val/RESULTS.md`; commit and push | laptop | |
| 15 | `prp freeze`; commit `prereg.lock.json`; push | laptop | seconds |
| 16 | `prp evaluate --split test --methods kmer_raw kmer_head plantcad_raw plantcad_head plantcad_ft2_head minimap2 minimap2_train` (the list in PREREG.md section 6; it checks every input exists before registering the run) | laptop | 10 min |
| 17 | copy `work/results/test/` to `results/`; commit `TEST_RUN.json` and results; push | laptop | |

Pushing after steps 14 and 15 leaves a public, timestamped record that the targets came before the
test run.

### Kaggle, step by step

1. kaggle.com > **Create > New Dataset** > upload `prp_windows.zip` (Windows Explorer can open the WSL
   path the packing script prints) > title `prp-windows` > **Private** > Create.
2. kaggle.com > **Create > New Notebook** > **File > Import Notebook** > upload
   `kaggle/plantcad_gpu.ipynb` from the repo.
3. Right-hand panel: **Accelerator: GPU T4 x2**, **Internet: On**, **Add Input** > your `prp-windows` dataset.
4. Run cells 1 to 5. Cell 5 (`gpu-check`) prints the GPU, the model's layer count, whether embeddings
   are identical for a read and its reverse complement, whether fp16 agrees with fp32, and an
   estimated time. It stops the notebook if anything is wrong.
5. If it says OK, run the remaining cells, or use **Save Version > Save & Run All** so the run
   continues with the browser closed (sessions are limited to about 12 hours; this run needs 2 to 4).
6. Download `gpu_outputs.zip` from the notebook's **Output** panel.

### Notes per step

- **resolve:** if a species prints `NO ASSEMBLY FOUND`, use the alternate given in its `note` in
  `configs/panel.yaml`, or pin an accession with `accession: GCF_...`, then re-run `prp check-panel --strict`.
- **simulate:** each Badread worker holds one region FASTA in memory (up to about 1 GB of sequence).
  Three workers suit 16 GB visible to WSL. `--backend builtin` is much faster but less realistic;
  if you use it, the write-up must say so.
- **gpu-check fails on fp16:** set `amp: none` under `encoders.plantcad` in `configs/experiment.yaml`
  (slower, full precision) before the long steps, and commit the change.
- **gpu-check fails on layer names:** copy its `suggested_patterns` into `finetune.trainable_regex`.
- **dev:** the only place you look at numbers before freezing. Change anything here, then re-run dev.
  Once you freeze, you stop.
- **evaluate --split test:** a second test run is refused unless you pass `--rerun-reason "..."`,
  which is recorded as a deviation.

## 4. If time is short

In priority order: `plantcad_head`, `kmer_head`, `minimap2` with `specimen_sizes: [1, 10]`; then add
`plantcad_ft2_head`; then `plantcad_raw`, `kmer_raw` and n = 50. Reducing `simulate.reads` to
`{train: 2000, val: 600, test: 600}` roughly halves simulation and embedding time. Make these
changes before freezing.

## 5. Troubleshooting

| symptom | fix |
|---|---|
| `datasets: command not found` | `conda install -c conda-forge ncbi-datasets-cli` |
| `badread not on PATH` | `pip install git+https://github.com/rrwick/Badread.git` |
| Kaggle setup cell runs for 20+ minutes | mamba-ssm is compiling from source; let it finish |
| `No GPU visible` on Kaggle | Accelerator is set to None or P100; choose GPU T4 x2 and restart |
| `cannot locate nucleotide tokens` | the tokenizer is not character-level; check the model id |
| `files changed since the preregistration was frozen` | you edited a frozen file; restore it from git |
