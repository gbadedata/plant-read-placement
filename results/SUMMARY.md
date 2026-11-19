# Results summary

One preregistered test run, on species the models never saw, after the targets were frozen.

- Targets pushed: 19 Nov 2026 02:12:50 UTC (commit `9b25b49`).
- Freeze pushed: 02:14:31 UTC (commit `5f83686`, `prereg.lock.json`).
- Test run: started 02:14:50 UTC, with no deviations (`TEST_RUN.json`).
- Results pushed: 02:28:38 UTC (commit `30fbd97`, `results/test/`).

The push times come from GitHub's activity log for the repository. Full tables are in
[`test/RESULTS.md`](test/RESULTS.md); the plan and targets are in [`../PREREG.md`](../PREREG.md).

## The preregistered hypotheses (test split, 10 reads per specimen)

A hypothesis counts as met only if the test point estimate reaches its target. Brackets show 95%
intervals from a species-level bootstrap.

| | Hypothesis | Validation | Target | Test | Met |
|---|---|---|---|---|---|
| H1 | `plantcad_head` balanced accuracy | 0.588 | >= 0.538 | 0.536 [0.437, 0.622] | no |
| H2 | `plantcad_head` minus `kmer_head` | +0.004 | >= +0.020 | -0.039 (`kmer_head` 0.575 [0.513, 0.637]) | no (predicted) |
| H3 | `plantcad_head` AUROC, new vs rest | 0.921 | >= 0.871 | 0.891 [0.775, 0.977] | **yes** |
| H4 | `plantcad_head` nearest genus right for relatives | 0.973 | >= 0.923 | 0.920 | no |
| H5 | `plantcad_ft2_head` minus `plantcad_head` | +0.011 | >= +0.020 | +0.018 (`plantcad_ft2_head` 0.554 [0.454, 0.640]) | no (predicted) |
| H6 | `minimap2` minus `plantcad_head` | +0.325 | <= +0.375 | +0.420 (`minimap2` 0.956 [0.930, 0.975]) | no |

One of six targets was met. H1, H4 and H5 missed by 0.003 or less, which is well inside their
intervals, but they are reported as missed because that was the rule. H2 and H5 were predicted in
writing, before the test, not to be met.

## What the results say

1. **The plant DNA language model did not beat 5-mer composition at the decision task.** With a
   trained head, PlantCaduceus scored 0.536 at 10 reads per specimen, against 0.575 for k-mers.
   - k-mers also held up better from validation to test: 0.584 to 0.575, against 0.588 to 0.536 for
     PlantCaduceus.
   - Without a head, the PlantCaduceus embeddings scored at chance (0.333).
   - All learned methods were close to chance on single reads (0.33 to 0.38).
2. **Alignment to full reference genomes was far ahead:** 0.956 at 10 reads and 0.834 on single reads.
   Its index contains the regions the test reads came from, which this benchmark stated in advance.
3. **The language model is better at recognising species than at deciding what kind of call to
   make.** Ignoring the thresholds, the nearest prototype was the right species for known specimens
   in 0.96 of cases at 50 reads, against 0.85 for k-mers. The model's weakness is the "relative"
   call (exploratory analysis below).
4. **Adapting the top four layers gave a small, uncertain gain.**
   - +0.018 at 10 reads, just under the preregistered bar.
   - +0.051 at 50 reads (0.677 against 0.626), a secondary outcome with overlapping intervals.
   - The first attempt collapsed; the development log in `PREREG.md` records the diagnosis.

## Exploratory analysis (not preregistered)

`prp explore --split test` re-reads the saved test scores to separate two failure modes. One is
the nearest prototype being wrong. The other is the distance to that prototype failing to separate
the classes.

| 50 reads per specimen | Known: nearest species right | Relatives: nearest genus right | Distance AUROC, known vs relative | Distance AUROC, relative vs new | Relatives called known / relative / new |
|---|---|---|---|---|---|
| `kmer_head` | 0.85 | 1.00 | 0.85 | 0.96 | 0.10 / 0.68 / 0.23 |
| `plantcad_head` | 0.96 | 0.96 | 0.94 | 0.80 | 0.34 / 0.09 / 0.57 |
| `plantcad_ft2_head` | 0.97 | 0.94 | 0.95 | 0.76 | 0.17 / 0.25 / 0.57 |
| `minimap2_train` | 0.77 | 1.00 | 0.57 | 1.00 | 0.00 / 0.15 / 0.85 |

- **PlantCaduceus puts relatives near the right genus, but the "relative" call fails.** The
  thresholds set on the validation species do not transfer to the test species.
  - On validation, most relatives fell inside the "known" threshold.
  - On test, most fell beyond the "new" threshold.
  - Relatives and new species overlap in distance (AUROC 0.80, against 0.96 for k-mers).
  - The language model's distance scale for "an unseen congener" varies from species to species
    more than composition does. A decision rule that uses more than one distance is the obvious
    next step.
- **The learned methods recognise species from unseen parts of a genome better than alignment
  does.** With an index of only the training regions, minimap2 picks the right species for 0.77 of
  known specimens, against 0.85 to 0.97 for the learned methods.
  - It never calls "known" correctly, because a known species' reads from unseen regions align only
    to repeats elsewhere in its genome.
  - Its thresholds, set on validation, did not transfer: 0.667 on validation, 0.506 on test.

## Limitations

- **The reads are simulated** (Badread) from real assemblies, not sequenced from specimens.
- **Few species per held-out class** (4 to 13), so the intervals are wide. Most differences between
  the learned methods are inside them.
- **The thresholds are tuned on the validation species.** That they transfer poorly is itself one
  of the findings.
- **The fine-tune's head scaled its inputs from only the first 128 training windows,** a design
  flaw recorded before the test. Both attempts used it.
- **The second deliverable in the brief,** flagging alkaloid-biosynthesis genes, was not attempted.
