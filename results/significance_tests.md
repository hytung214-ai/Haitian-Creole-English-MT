# Pelle Converged Run Significance Tests

Date created: 2026-05-25

Source directory: `experiments/fairseq_nmt_converged`

Test set: `official_test`

Baseline system: `full`

Method: paired bootstrap resampling with sacreBLEU 2.6.0.

Number of resamples: 10,000.

## BLEU Paired Bootstrap

| comparison against full | BLEU | delta vs full | p-value | significant at p < 0.05 |
|---|---:|---:|---:|---|
| full | 43.03 | 0.00 | n/a | baseline |
| light_control | 42.40 | -0.64 | 0.036996 | yes |
| light_clean | 41.86 | -1.18 | 0.000300 | yes |
| risk_clean | 41.57 | -1.47 | 0.000600 | yes |
| risk_control | 39.97 | -3.06 | 0.000100 | yes |

## chrF2 Paired Bootstrap

| comparison against full | chrF2 | delta vs full | p-value | significant at p < 0.05 |
|---|---:|---:|---:|---|
| full | 64.8 | 0.0 | n/a | baseline |
| light_control | 64.4 | -0.4 | 0.0382 | yes |
| light_clean | 63.9 | -0.9 | 0.0004 | yes |
| risk_clean | 63.5 | -1.3 | 0.0002 | yes |
| risk_control | 62.5 | -2.3 | 0.0001 | yes |

## Interpretation

Using paired bootstrap resampling on the official test set, the full-data model
is significantly different from each cleaned-data system at `p < 0.05` for both
BLEU and chrF2. Since the full-data model also has the highest score in every
comparison, this supports the claim that the full-data system performs better
than the cleaned-data systems on the official test set.

The smallest difference is between `full` and `light_control`: 0.64 BLEU and
0.4 chrF2. This comparison is statistically significant but close to the
threshold, so it should be described cautiously. The risk-aware systems show
larger and more clearly significant drops.

## Command

BLEU:

```bash
sacrebleu \
  experiments/fairseq_nmt_converged/full/generate/official_test/ref.detok \
  -i \
  experiments/fairseq_nmt_converged/full/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/light_control/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/light_clean/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/risk_clean/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/risk_control/generate/official_test/hyp.detok \
  -m bleu --paired-bs --paired-bs-n 10000 --paired-jobs 4
```

chrF2:

```bash
sacrebleu \
  experiments/fairseq_nmt_converged/full/generate/official_test/ref.detok \
  -i \
  experiments/fairseq_nmt_converged/full/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/light_control/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/light_clean/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/risk_clean/generate/official_test/hyp.detok \
  experiments/fairseq_nmt_converged/risk_control/generate/official_test/hyp.detok \
  -m chrf --paired-bs --paired-bs-n 10000 --paired-jobs 4
```

## Note

The paired bootstrap test evaluates whether the observed score difference is
likely to be due to chance under paired resampling of the same test sentences.
It does not by itself explain why one system is better; interpretation should be
combined with the data-cleaning analysis and qualitative output inspection.
