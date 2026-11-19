# Preregistration: placing plant specimens from noisy Nanopore reads

**Status: targets set from the validation run on 19 Nov 2026; frozen with `prp freeze` (see
`prereg.lock.json`).** The freeze command refuses while any target placeholder remains. After freezing, this file, the configs,
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

## 6. Targets (set from validation before test)

Rule, fixed before any test specimen was scored. Single scores (H1, H3, H4): the validation value minus
0.05, because the test species are different species and the thresholds were tuned on the validation
specimens. Differences that ask "does it add anything" (H2, H5): at least +0.02, the smallest gain that
would be called an improvement; "slightly below the validation value" would put these near zero, so that
+0.001 would count. Neither reaches +0.02 on validation, so both are predicted not to be met.
minimap2's lead (H6): the validation lead plus 0.05. Validation values are at 10 reads per specimen,
with species-level bootstrap 95% intervals where the metric has one.

| | hypothesis | validation value | target on test |
|---|---|---|---|
| H1 | `plantcad_head` balanced accuracy at n = 10 | 0.588 (0.506 to 0.665) | >= 0.538 |
| H2 | `plantcad_head` minus `kmer_head` balanced accuracy at n = 10 (does the language model add anything over composition?) | +0.004 (`kmer_head` 0.584, 0.525 to 0.637) | >= +0.020 (predicted not met) |
| H3 | `plantcad_head` AUROC new vs rest at n = 10 | 0.921 (`kmer_head` 0.940) | >= 0.871 |
| H4 | `plantcad_head` relative-genus rate at n = 10: `relative_genus_top1`, the share of relative specimens whose nearest prototype is in the right genus, ignoring the thresholds | 0.973 | >= 0.923 |
| H5 | `plantcad_ft2_head` minus `plantcad_head` balanced accuracy at n = 10 (does adapting the backbone help?) | +0.011 (`plantcad_ft2_head` 0.598, 0.547 to 0.645) | >= +0.020 (predicted not met) |
| H6 | `minimap2` minus `plantcad_head` balanced accuracy at n = 10 (prediction stated in advance; minimap2 is expected to lead on known species because it indexes their full genomes) | +0.325 (`minimap2` 0.913, 0.779 to 0.987) | <= +0.375 |

Evaluated once on test: `kmer_raw`, `kmer_head`, `plantcad_raw`, `plantcad_head`, `plantcad_ft2_head`,
`minimap2`, and `minimap2_train` as a reference row with no hypothesis. `plantcad_ft_head` (fine-tune
attempt 1) is a validation record only.

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
- **Diagnosis of attempt 1, written 18 Nov 2026 about 23:00, before attempt 2 finished.** The loss splits
  into species, genus and family terms; all three sat at 3.93 from step 250 to step 3000. ln(51) = 3.93 is
  exactly what this loss gives when every pair in a 52-read batch has the same similarity, i.e. the
  head inside the fine-tune collapsed: it stopped separating anything, rather than learning slowly. The
  total, 1.75 x ln(51) = 6.88, is also where attempt 2 sits at steps 250 and 500 (6.92, 6.87), so attempt 2
  is predicted to end the same way. Two likely causes, neither tested yet: (a) little signal per step
  (1 or 3 windows of 512 bp per read, 52 reads); (b) a design flaw: that head standardises its inputs with
  statistics from only the first 128 training windows (about 18 reads of one species), while PlantCaduceus
  read embeddings are extremely anisotropic (median cosine 0.998 between two random reads; 99% of the
  variance in 5 of 384 directions), so species information sits in tiny directions that those statistics
  mis-scale. The frozen-model head, standardised over all training reads, did not collapse. Correction to
  attempt 1's entry: the adapted encoder's outputs did change (mostly a constant offset, plus a per-read
  change about one sixth of the typical distance between two reads); the change did not help placement.
  The commitment stands: no attempt 3 before the test.
- **Attempt 2 outcome against the prediction above (19 Nov 2026).** Half right. Attempt 2 started at the
  collapsed level (each term 3.89 at step 250) but slowly moved below it (3.82 by step 3000, total 6.70
  against 6.88 collapsed and 7.54 untrained), so it learned a little, which attempt 1 never did. 3 of
  3,000 updates were skipped and the trainable weights moved 1.8%, which rules out skipped updates and
  frozen weights as causes. Downstream it adds little: a head on its features scored 0.565 during
  training against 0.569 for the frozen model, and +0.011 balanced accuracy on validation at n = 10.
  The frozen embeddings were re-made in that Kaggle session and matched the first session's exactly
  (maximum difference 0 on all splits).
- **H4 wording clarified (19 Nov 2026, before freezing).** "Relative-genus rate" could mean the
  end-to-end relative recall or the threshold-free nearest-genus accuracy. It is the latter,
  `relative_genus_top1`, the name the training monitor and the secondary outcomes already use
  ("nearest-genus accuracy for relatives"). End-to-end relative recall is reported for every method
  anyway (validation, n = 10: `plantcad_head` 0.43, `kmer_head` 0.71).
- **`minimap2_train` added (18 Nov 2026), after seeing the validation table.** `minimap2` indexes the full
  panel genomes, so a known species' validation and test reads align to the exact sequence they came
  from; H6 and section 8 already said this favours it. `minimap2_train` indexes only the training blocks,
  matching what the learned methods saw. It is a reference row with no hypothesis and cannot change
  any H1-H6 outcome. Both rows are reported: `minimap2` for "a reference genome exists", `minimap2_train`
  for "like-for-like with the learned methods".

## 10. Deviations

Recorded automatically in `TEST_RUN.json` and summarised in the results.
