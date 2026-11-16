# Pelle Converged Run Results

Date created: 2026-05-25

Source directory: `experiments/fairseq_nmt_converged`

This report summarizes the final converged fairseq NMT runs synced from Pelle.
It includes automatic evaluation scores and the final validation-loss /
perplexity indicators from each training log.

## Final Results

| condition | internal BLEU | internal chrF2 | official BLEU | official chrF2 | final valid loss | best valid loss | final valid ppl | updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 60.5 | 75.1 | 43.0 | 64.8 | 2.930 | 2.909 | 2.34 | 200000 |
| light_clean | 58.5 | 73.9 | 41.9 | 63.9 | 3.095 | 3.089 | 2.65 | 113511 |
| light_control | 59.8 | 74.7 | 42.4 | 64.4 | 2.964 | 2.957 | 2.40 | 160000 |
| risk_clean | 55.3 | 71.9 | 41.6 | 63.5 | 3.376 | 3.367 | 3.25 | 63738 |
| risk_control | 56.9 | 73.0 | 40.0 | 62.5 | 3.201 | 3.193 | 2.81 | 77000 |

## Main Finding

The full-data model is the best final system on both test sets. It reaches
60.5 BLEU on the internal test set and 43.0 BLEU on the official test set.
The best cleaned-data comparison is `light_control`, with 42.4 official BLEU,
but it still remains below the full-data model.

The risk-aware conditions perform worse than the full and light-cleaning
conditions. This suggests that the aggressive/risk-aware filtering removed
training examples that still contributed useful lexical or domain coverage.

## Statistical Significance

Paired bootstrap resampling was run on the official test set with `full` as the
baseline system, using sacreBLEU 2.6.0 and 10,000 bootstrap resamples.

| comparison against full | BLEU delta | BLEU p-value | chrF2 delta | chrF2 p-value | significant at p < 0.05 |
|---|---:|---:|---:|---:|---|
| light_control | -0.64 | 0.036996 | -0.4 | 0.0382 | yes |
| light_clean | -1.18 | 0.000300 | -0.9 | 0.0004 | yes |
| risk_clean | -1.47 | 0.000600 | -1.3 | 0.0002 | yes |
| risk_control | -3.06 | 0.000100 | -2.3 | 0.0001 | yes |

The `full` system is significantly different from every cleaned-data system on
the official test set for both BLEU and chrF2. Because `full` also has the
highest score in every comparison, this supports the claim that the full-data
model performs better than the cleaned-data models in the final converged
setting. The `full` vs. `light_control` comparison is the closest and should be
described cautiously, since its p-values are close to 0.05.

Detailed significance-test notes are saved in
`reports/experiment_results/converged_significance_tests.md`.

## Convergence Notes

The full-data run reached the maximum training budget of 200k updates. The
other conditions stopped earlier, consistent with patience-based stopping after
validation loss plateaued.

For all conditions, the final validation loss is very close to the best
validation loss. This indicates that the models were evaluated near their
validation-loss plateau rather than during an unstable training phase.


