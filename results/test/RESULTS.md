# Plant read placement (test split)

Test split first evaluated 2026-11-19T02:14:50+00:00; deviations recorded: 0.

## Three-way placement

Balanced accuracy is the mean of the three per-class rates. A known specimen counts as correct only if it is called known *and* assigned the right species; a relative only if called relative *and* given the right genus; a new specimen if called new. Intervals resample species within each class.

| method            |   reads per specimen | balanced accuracy [95% CI]   |   known (species right) |   relative (genus right) |   new |   AUROC new vs rest |   specimens |
|:------------------|---------------------:|:-----------------------------|------------------------:|-------------------------:|------:|--------------------:|------------:|
| kmer_head         |                    1 | 0.377 [0.357, 0.397]         |                   0.181 |                    0.081 | 0.868 |               0.648 |        4400 |
| kmer_raw          |                    1 | 0.331 [0.314, 0.345]         |                   0.025 |                    0.011 | 0.958 |               0.623 |        4400 |
| minimap2          |                    1 | 0.834 [0.790, 0.872]         |                   0.9   |                    0.704 | 0.898 |               0.967 |        4400 |
| minimap2_train    |                    1 | 0.525 [0.476, 0.572]         |                   0     |                    0.671 | 0.903 |               0.879 |        4400 |
| plantcad_ft2_head |                    1 | 0.333 [0.294, 0.372]         |                   0     |                    0.228 | 0.772 |               0.605 |        4400 |
| plantcad_head     |                    1 | 0.337 [0.308, 0.364]         |                   0     |                    0.092 | 0.919 |               0.612 |        4400 |
| plantcad_raw      |                    1 | 0.333 [0.333, 0.333]         |                   0     |                    0     | 1     |               0.541 |        4400 |
| kmer_head         |                   10 | 0.575 [0.513, 0.637]         |                   0.239 |                    0.667 | 0.818 |               0.933 |        2200 |
| kmer_raw          |                   10 | 0.409 [0.333, 0.480]         |                   0.21  |                    0.203 | 0.816 |               0.773 |        2200 |
| minimap2          |                   10 | 0.956 [0.930, 0.975]         |                   0.936 |                    0.973 | 0.96  |               0.999 |        2200 |
| minimap2_train    |                   10 | 0.506 [0.430, 0.608]         |                   0     |                    0.517 | 1     |               0.99  |        2200 |
| plantcad_ft2_head |                   10 | 0.554 [0.454, 0.640]         |                   0.565 |                    0.28  | 0.818 |               0.888 |        2200 |
| plantcad_head     |                   10 | 0.536 [0.437, 0.622]         |                   0.462 |                    0.323 | 0.824 |               0.891 |        2200 |
| plantcad_raw      |                   10 | 0.332 [0.269, 0.393]         |                   0.177 |                    0.037 | 0.782 |               0.59  |        2200 |
| kmer_head         |                   50 | 0.659 [0.595, 0.727]         |                   0.373 |                    0.675 | 0.93  |               0.989 |         440 |
| kmer_raw          |                   50 | 0.544 [0.440, 0.659]         |                   0.431 |                    0.25  | 0.95  |               0.91  |         440 |
| minimap2          |                   50 | 0.980 [0.957, 0.995]         |                   0.969 |                    1     | 0.97  |               1     |         440 |
| minimap2_train    |                   50 | 0.383 [0.333, 0.483]         |                   0     |                    0.15  | 1     |               0.999 |         440 |
| plantcad_ft2_head |                   50 | 0.677 [0.570, 0.769]         |                   0.912 |                    0.25  | 0.87  |               0.938 |         440 |
| plantcad_head     |                   50 | 0.626 [0.532, 0.691]         |                   0.919 |                    0.087 | 0.87  |               0.945 |         440 |
| plantcad_raw      |                   50 | 0.428 [0.298, 0.550]         |                   0.665 |                    0     | 0.62  |               0.735 |         440 |

## Single reads by read identity

| method            | identity bin   |   reads |   balanced accuracy |
|:------------------|:---------------|--------:|--------------------:|
| kmer_raw          | [0, 0.88)      |     439 |               0.33  |
| kmer_raw          | [0.88, 0.92)   |    1495 |               0.335 |
| kmer_raw          | [0.92, 0.96)   |    2227 |               0.329 |
| kmer_raw          | [0.96, 1)      |     239 |               0.334 |
| kmer_head         | [0, 0.88)      |     439 |               0.355 |
| kmer_head         | [0.88, 0.92)   |    1495 |               0.374 |
| kmer_head         | [0.92, 0.96)   |    2227 |               0.382 |
| kmer_head         | [0.96, 1)      |     239 |               0.384 |
| plantcad_raw      | [0, 0.88)      |     439 |               0.333 |
| plantcad_raw      | [0.88, 0.92)   |    1495 |               0.333 |
| plantcad_raw      | [0.92, 0.96)   |    2227 |               0.333 |
| plantcad_raw      | [0.96, 1)      |     239 |               0.333 |
| plantcad_head     | [0, 0.88)      |     439 |               0.353 |
| plantcad_head     | [0.88, 0.92)   |    1495 |               0.331 |
| plantcad_head     | [0.92, 0.96)   |    2227 |               0.338 |
| plantcad_head     | [0.96, 1)      |     239 |               0.347 |
| plantcad_ft2_head | [0, 0.88)      |     439 |               0.362 |
| plantcad_ft2_head | [0.88, 0.92)   |    1495 |               0.328 |
| plantcad_ft2_head | [0.92, 0.96)   |    2227 |               0.335 |
| plantcad_ft2_head | [0.96, 1)      |     239 |               0.3   |
| minimap2          | [0, 0.88)      |     439 |               0.654 |
| minimap2          | [0.88, 0.92)   |    1495 |               0.857 |
| minimap2          | [0.92, 0.96)   |    2227 |               0.85  |
| minimap2          | [0.96, 1)      |     239 |               0.887 |
| minimap2_train    | [0, 0.88)      |     439 |               0.488 |
| minimap2_train    | [0.88, 0.92)   |    1495 |               0.527 |
| minimap2_train    | [0.92, 0.96)   |    2227 |               0.527 |
| minimap2_train    | [0.96, 1)      |     239 |               0.562 |
