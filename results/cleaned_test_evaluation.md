# Cleaned Official Test Evaluation Preview

Date created: 2026-05-27

Source model directory: `experiments/fairseq_nmt_converged`

Cleaned test data directory: `data/official_test_cleaned`

Evaluation output directory:
`experiments/fairseq_nmt_converged_cleaned_official_test_eval`

This report summarizes an additional evaluation where each cleaned-data model is
tested on an official test set filtered by the corresponding cleaning rule:

- `light_clean` model evaluated on `light_clean`-filtered official test data.
- `risk_clean` model evaluated on `risk_clean`-filtered official test data.

## Cleaned Test Data

| cleaned test set | original pairs | kept pairs | removed pairs | removed percent |
|---|---:|---:|---:|---:|
| light_clean official_test | 1228 | 1220 | 8 | 0.65 |
| risk_clean official_test | 1228 | 1012 | 216 | 17.59 |

The `light_clean` filter removes only a small number of official-test examples,
mainly URL/email, extreme length-ratio, and very long segments. The `risk_clean`
filter is much more aggressive, removing title-like fragments, product/technical
segments, web-navigation-like examples, and other suspicious alignments.

Detailed cleaning records are saved in:

- `reports/data_preparation/official_test_cleaning/light_clean_removed.tsv`
- `reports/data_preparation/official_test_cleaning/risk_clean_removed.tsv`
- `reports/data_preparation/official_test_cleaning/light_clean_kept_map.tsv`
- `reports/data_preparation/official_test_cleaning/risk_clean_kept_map.tsv`

## Results

| model | test set | test pairs | BLEU | chrF2 | original official BLEU | original official chrF2 | BLEU change | chrF2 change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| light_clean | light_clean-filtered official_test | 1220 | 41.5 | 63.8 | 41.9 | 63.9 | -0.4 | -0.1 |
| risk_clean | risk_clean-filtered official_test | 1012 | 41.0 | 63.3 | 41.6 | 63.5 | -0.6 | -0.2 |

## Confidence Intervals

Bootstrap confidence intervals were computed with sacreBLEU 2.6.0 using 10,000
bootstrap resamples and seed 12345.

| model | test set | test pairs | BLEU | BLEU confidence | chrF2 | chrF2 confidence |
|---|---|---:|---:|---|---:|---|
| light_clean | light_clean-filtered official_test | 1220 | 41.5 | μ = 41.5 ± 2.0 | 63.8 | μ = 63.8 ± 1.5 |
| risk_clean | risk_clean-filtered official_test | 1012 | 41.0 | μ = 41.0 ± 2.1 | 63.3 | μ = 63.3 ± 1.5 |

These intervals describe uncertainty for each score on its own cleaned test
subset. They should not be interpreted as a paired significance comparison
between `light_clean` and `risk_clean`, because the two cleaned test sets contain
different sentence subsets.

## Main Finding

Applying the corresponding cleaning rule to the official test set does not
improve the cleaned-data models' evaluation scores. Both models score slightly
lower on their cleaned official-test subsets than on the original official test
set.

This suggests that the lower performance of the cleaned-data models is not
explained by a mismatch where noisy official-test examples unfairly hurt the
cleaned systems. Instead, the result is consistent with the earlier
interpretation that filtering, especially risk-aware filtering, may remove
training examples that still provide useful lexical, domain, or stylistic
coverage.

## Output Files

Light-clean evaluation:

- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/light_clean_on_light_clean_official_test/generate/generate.out`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/light_clean_on_light_clean_official_test/generate/hyp.detok`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/light_clean_on_light_clean_official_test/generate/ref.detok`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/light_clean_on_light_clean_official_test/generate/scores.txt`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/light_clean_on_light_clean_official_test/generate/confidence.txt`

Risk-clean evaluation:

- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/risk_clean_on_risk_clean_official_test/generate/generate.out`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/risk_clean_on_risk_clean_official_test/generate/hyp.detok`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/risk_clean_on_risk_clean_official_test/generate/ref.detok`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/risk_clean_on_risk_clean_official_test/generate/scores.txt`
- `experiments/fairseq_nmt_converged_cleaned_official_test_eval/risk_clean_on_risk_clean_official_test/generate/confidence.txt`

## Reproducibility

The cleaned official-test data was created with:

```bash
python scripts/clean_parallel_data.py \
  --src data/official_test/test.hat \
  --tgt data/official_test/test.eng \
  --output-dir data/official_test_cleaned \
  --report-dir reports/data_preparation/official_test_cleaning
```

The evaluations were run with:

```bash
OVERWRITE=1 USE_CPU=0 CONDITION=light_clean TEST_CONDITION=light_clean \
  scripts/run_cleaned_official_test_eval.sh

OVERWRITE=1 USE_CPU=0 CONDITION=risk_clean TEST_CONDITION=risk_clean \
  scripts/run_cleaned_official_test_eval.sh
```

The confidence intervals were computed with:

```bash
sacrebleu \
  experiments/fairseq_nmt_converged_cleaned_official_test_eval/light_clean_on_light_clean_official_test/generate/ref.detok \
  -i experiments/fairseq_nmt_converged_cleaned_official_test_eval/light_clean_on_light_clean_official_test/generate/hyp.detok \
  -m bleu chrf --confidence --confidence-n 10000

sacrebleu \
  experiments/fairseq_nmt_converged_cleaned_official_test_eval/risk_clean_on_risk_clean_official_test/generate/ref.detok \
  -i experiments/fairseq_nmt_converged_cleaned_official_test_eval/risk_clean_on_risk_clean_official_test/generate/hyp.detok \
  -m bleu chrf --confidence --confidence-n 10000
```
