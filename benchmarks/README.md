# Benchmarks

Scripts for the experiments reported in the article. All of them read the
normalized registries in `data/normalized/` (institutional data, not in git)
and write to `data/normalized/results/`.

| Script | Experiment |
|---|---|
| `maxent_reference.py` | MaxEnt-equivalent module vs a full MaxEnt implementation (elapid) on identical presence, background, covariates, random splits and spatial-block folds |
| `satscan_export.py` / `satscan_compare.py` | Space–time scan module vs SaTScan on identical district-year input |
| `reuse_fifth_disease.py` | A fifth disease (foot-and-mouth disease) through the unchanged pipeline, with disease-specific vs shared code accounting |

```bash
pip install -r requirements-benchmarks.txt
python3 benchmarks/maxent_reference.py --disease anthrax
python3 benchmarks/maxent_reference.py --disease rabies --reservoir-category livestock
python3 benchmarks/maxent_reference.py --disease rabies --reservoir-category companion

python3 run_space_time_scan.py --disease anthrax --n-permutations 999
python3 benchmarks/satscan_export.py --disease anthrax
#   -> run SaTScan with data/normalized/results/satscan/anthrax/SETTINGS.txt
python3 benchmarks/satscan_compare.py --disease anthrax

python3 benchmarks/reuse_fifth_disease.py --disease fmd
```
