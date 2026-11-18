# Preregistration: placing plant specimens from noisy Nanopore reads

**Status: draft.** Every `TBD` below must be replaced with a number from the validation run before
`prp freeze`; the freeze command refuses while any remain. After freezing, this file, the configs,
the calibrated thresholds and the trained weights are hashed into `prereg.lock.json`, and the test
split can be scored once. Any re-run is logged as a deviation in `TEST_RUN.json`.

## 1. Question

Given only noisy, low-coverage Oxford Nanopore reads from a plant specimen, and no reference genome
for that specimen, can a plant DNA language model adapted with a contrastive head decide whether the
specimen is a **known species** (and which), an **unseen relative** of a known genus (and which
genus), or **likely new** (no congener in the reference panel)? How does it compare with an
alignment baseline (minimap2) and an alignment-free composition baseline (k-mers)?

## 2. Data (fixed before any model was trained)

- Species and roles: `configs/panel.yaml` (13 panel species; 3 + 3 validation and 4 + 5 test
  held-out species). No species from Poaceae or Brassicales, the two clades PlantCaduceus was
  pre-trained on, is held out.
- Genome assemblies: pinned by accession and SHA-256 in `work/genomes/lock.json` (frozen).
- Regions: panel genomes cut into 1 Mb blocks, assigned to train/val/test (80/10/10) by a seeded
  hash; held-out species contribute only to their own split. Sequence is upper-cased before reads are
  simulated, so assembly soft-masking cannot leak into reads.
- Reads: Badread 0.4.2, `nanopore2023` error and quality models, identity beta(mean 92%, max 98%,
  sd 3%), length mean 8 kb (sd 6 kb), default junk, random and chimeric reads. Junk/random reads are
  excluded from training labels and kept in evaluation.
- Specimens: 1, 10 or 50 reads drawn without replacement from one species and split.

## 3. Methods compared (fixed list)

| id | features | trained part |
|---|---|---|
| `plantcad_head` | PlantCaduceus l20, frozen, strand-averaged, mean over 512 bp windows | contrastive head |
| `plantcad_ft2_head` | PlantCaduceus l20 with top blocks adapted on noisy read windows (attempt 2; see development log) | contrastive head |
| `plantcad_raw` | PlantCaduceus l20, frozen | none (ablation) |
| `kmer_head` | canonical 5-mer frequencies | contrastive head (ablation: same head, no language model) |
| `kmer_raw` | canonical 5-mer frequencies | none |
| `minimap2` | best alignment of each read to every panel genome (full genomes, test regions included) | none |
| `minimap2_train` | best alignment to the training blocks of every panel genome only, the same sequence the learned methods saw (reference row, no hypothesis; see development log) | none |

Contrastive head: supervised contrastive loss at species level plus genus (weight 0.5) and family
(weight 0.25) terms, trained on panel species' training-region reads only. The checkpoint with the
best validation score is kept.

If `plantcad_ft2_head` has not finished before freezing, it is reported as not run, not dropped silently.
`plantcad_ft_head` (fine-tune attempt 1) is reported on validation as a record and is not evaluated on test.

## 4. Decision rule and calibration (fixed)

Specimen embedding = normalised mean of its read embeddings; distance d1 = 1 - cosine similarity to
the nearest panel-species prototype (minimap2: 1 - mean matched-base fraction). Call **known** if
d1 <= t1, **relative** if t1 < d1 <= t2, otherwise **new**. t1 and t2 are chosen per method and per
specimen size on validation specimens only, to maximise balanced accuracy.

A call counts as correct only end to end: known needs the right species, relative needs the right
genus, new needs the call "new".

## 5. Outcomes

- **Primary:** balanced accuracy (mean of the three per-class rates) at **10 reads per specimen** on
  the test split, with a 95% interval from a species-level bootstrap (2,000 resamples).
- **Secondary:** the same at 1 and 50 reads; per-class rates; AUROC of d1 for new versus the rest;
  nearest-species accuracy for known specimens; nearest-genus accuracy for relatives; single-read
  accuracy by read-identity bin.

## 6. Targets (set from validation before test; fill every TBD)

Guidance: set each target slightly below the validation value, because the test species are
different species and some shrinkage is expected. Write the validation value next to each target.

| | hypothesis | validation value | target on test |
|---|---|---|---|
| H1 | `plantcad_head` balanced accuracy at n = 10 | TBD | >= TBD |
| H2 | `plantcad_head` minus `kmer_head` balanced accuracy at n = 10 (does the language model add anything over composition?) | TBD | >= TBD |
| H3 | `plantcad_head` AUROC new vs rest at n = 10 | TBD | >= TBD |
| H4 | `plantcad_head` relative-genus rate at n = 10 | TBD | >= TBD |
| H5 | `plantcad_ft2_head` minus `plantcad_head` balanced accuracy at n = 10 (does adapting the backbone help?) | TBD | >= TBD |
| H6 | `minimap2` minus `plantcad_head` balanced accuracy at n = 10 (prediction stated in advance; minimap2 is expected to lead on known species because it indexes their full genomes) | TBD | <= TBD |

A hypothesis is met if the test point estimate reaches its target. With 4 to 13 species per class
the intervals will be wide; they are reported for every outcome and no claim is made that an interval
excluding zero would not support.

## 7. Analyses not planned here

Anything not listed above, including any new method, threshold rule or specimen size, is exploratory
and will be labelled as such.

## 8. Known limitations (stated before seeing test results)

- Reads are simulated by one simulator; real Nanopore data from field specimens will differ
  (contamination, degraded DNA, non-plant reads).
- PlantCaduceus was pre-trained mainly on gene windows with repeats down-weighted, while
  low-coverage whole-genome reads are mostly repeats and intergenic sequence: a domain shift.
- The model sees 512 bp windows; long reads are summarised by averaging up to eight windows.
- minimap2's index contains the full genomes of panel species, including test regions; this favours
  the baseline on known species and is realistic for a reference database.
- Few held-out species per class, so species-level intervals are wide.
- The second deliverable in the brief (flagging alkaloid-biosynthesis genes) is not attempted here.

## 9. Development log (before freezing)

Changes made on validation before the lock, kept here so they are visible next to the plan.

- **Fine-tune attempt 1 (18 Nov 2026).** Top four blocks and the final norm, 1 window per read,
  4 reads per species (52 windows per step), backbone learning rate 2e-5, 3,000 steps, fp16 on a T4.
  The first configuration (4 windows per read, 208 per step) ran out of GPU memory and was reduced
  to this before any step completed. Result: training loss fell from 7.55 (random embeddings on the
  same labels) to about 6.85 by step 250 and stayed there; features from the adapted encoder gave the
  same validation score as the frozen model (best 0.569 for both heads). The encoder did not adapt
  measurably, so this run says nothing about whether adapting helps.
- **Fine-tune attempt 2: the only retry.** Settings fixed before it runs: 3 windows per read (156 per
  step), backbone learning rate 1e-4, everything else unchanged. Added logging of skipped fp16 updates,
  encoder gradient norm and how far the weights move. Whatever attempt 2 gives is the `plantcad_ft2_head`
  result; there is no attempt 3 before the test.
- **`minimap2_train` added (18 Nov 2026), after seeing the validation table.** `minimap2` indexes the full
  panel genomes, so a known species' validation and test reads align to the exact sequence they came
  from; H6 and section 8 already said this favours it. `minimap2_train` indexes only the training blocks,
  matching what the learned methods saw. It is a reference row with no hypothesis and cannot change
  any H1-H6 outcome. Both rows are reported: `minimap2` for "a reference genome exists", `minimap2_train`
  for "like-for-like with the learned methods".

## 10. Deviations

Recorded automatically in `TEST_RUN.json` and summarised in the results.
