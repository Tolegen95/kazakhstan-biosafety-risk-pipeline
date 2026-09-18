# Literature-derived data

Unlike `data/raw/` (institutional veterinary registries, gitignored, access
restricted per the article's Data Availability Statement), the two CSVs here
are reconstructed from numbers **already published** by the same author
group, so they are tracked in git.

- `cryptosporidiosis_strata.csv` — from Ussenbayev, A., Kurenkeyeva, D.,
  Bauer, C., Kadyrov, A. (2020). "Prevalence of Calves' Cryptosporidiosis in
  Northern Kazakhstan." *ICCSA 2020*, LNCS 12253, pp. 718-726. Overall and
  by-sex counts (n, y) are stated directly in the text. By-age-group counts
  are **reconstructed**: the paper's Table 1 gives only the MLE prevalence
  and n per age group, not the raw positive count, so `n_positive` here is
  `round(MLE * n)`. This introduces up to ±1 rounding discrepancy against
  the overall total (reconstructed age-group total: 36; paper's overall
  total: 35) — expected, not an error in this file.
  One correction was necessary: the paper's Methods text states n=344 for
  the 4-12 month age group, which is internally inconsistent (257+303+344 =
  904 ≠ 894, the stated total). n=334 is used instead (257+303+334 = 894,
  exactly matching the total), and 334 also gives a closer round-trip to the
  reported MLE (8/334=0.0240 vs. reported 0.0244) than 344 would (8/344=0.0233).

  Reproduction quality differs by stratum, and this is expected, not a bug:
  running `run_literature_modules.py` with the prior fitted to the paper's
  *overall* posterior reproduces the overall and by-sex posteriors exactly
  (their n_positive is a value stated directly in the paper's text), but
  only approximates the by-age-group posteriors (their n_positive is
  reconstructed from a 4-decimal MLE, and the paper may have used a
  different, more informative prior specifically for that breakdown, which
  cannot be recovered from the published numbers alone).

- `calf_diarrhea_prevalence.csv` — from Kadyrov, A., Ussenbayev, A.,
  Kurenkeyeva, D., Kurenkey, B., Abdrakhmanov, S., Tashatov, N. (2023).
  "Principal component analysis in the epidemiology of diarrhoea in
  calves." *Indonesian Journal of Electrical Engineering and Computer
  Science*, 30(3), 1762-1770. Open access, CC BY-SA. Raw counts taken
  directly from the paper's Table 1 (by age group) and Table 2 (by farm
  size class) -- no reconstruction needed.

Both are used to validate the Bayesian beta-binomial and PCA modules against
already-peer-reviewed results, without waiting on new data collection.
