# Train/Test Contamination Summary

Date created: 2026-05-27

Training data checked:

- `data/full/train.hat`
- `data/full/train.eng`

Test sets checked:

- `data/official_test/test.hat`, `data/official_test/test.eng`
- `data/internal_test/test.hat`, `data/internal_test/test.eng`

This report combines the train/test overlap checks for the official and
internal test sets. We checked exact and normalized overlap for parallel sentence
pairs, source-side sentences, and target-side sentences.

Normalization lowercases text, unescapes HTML entities, trims edge whitespace,
and collapses repeated whitespace.

## Overlap Counts

| test set | test pairs | overlap type | matched test lines | train occurrences | share of test set |
|---|---:|---|---:|---:|---:|
| official | 1228 | exact parallel pair | 4 | 4 | 0.33% |
| official | 1228 | normalized parallel pair | 13 | 13 | 1.06% |
| official | 1228 | exact source | 57 | 248 | 4.64% |
| official | 1228 | normalized source | 61 | 254 | 4.97% |
| official | 1228 | exact target | 49 | 433 | 3.99% |
| official | 1228 | normalized target | 57 | 477 | 4.64% |
| internal | 66503 | exact parallel pair | 0 | 0 | 0.00% |
| internal | 66503 | normalized parallel pair | 5003 | 5219 | 7.52% |
| internal | 66503 | exact source | 10234 | 100695 | 15.39% |
| internal | 66503 | normalized source | 14109 | 119013 | 21.22% |
| internal | 66503 | exact target | 7319 | 69743 | 11.01% |
| internal | 66503 | normalized target | 11547 | 82570 | 17.36% |

## Score Impact After Removing Overlap

The table below reports the full-data model score before and after removing
overlapping test examples. The filtered subsets remove test examples whose
normalized pair, source, or target appears in `data/full/train.*`.

| test set | evaluation subset | kept test lines | BLEU | BLEU change | chrF2 | chrF2 change |
|---|---|---:|---:|---:|---:|---:|
| official | original | 1228 | 43.0 | 0.0 | 64.8 | 0.0 |
| official | excluding normalized pair overlap | 1215 | 43.0 | 0.0 | 64.8 | 0.0 |
| official | excluding normalized source overlap | 1167 | 43.1 | +0.1 | 65.1 | +0.3 |
| official | excluding normalized target overlap | 1171 | 43.1 | +0.1 | 65.0 | +0.2 |
| internal | original | 66503 | 60.5 | 0.0 | 75.1 | 0.0 |
| internal | excluding normalized pair overlap | 61500 | 59.2 | -1.3 | 74.5 | -0.6 |
| internal | excluding normalized source overlap | 52394 | 60.2 | -0.3 | 76.0 | +0.9 |
| internal | excluding normalized target overlap | 54956 | 59.6 | -0.9 | 75.3 | +0.2 |

## Interpretation

The official test set has only minor overlap with the full training set. The
strictest signal, normalized parallel-pair overlap, affects 13 examples, or
1.06% of the official test set. Removing these examples does not change the
full model's official-test score: it remains 43.0 BLEU and 64.8 chrF2. This
suggests that exact or normalized train/test sentence-pair contamination does
not explain the full model's official-test BLEU.

The internal test set is different. It has no exact parallel-pair overlap with
the full training set, confirming that the split did not duplicate identical raw
lines into both train and internal test. However, after normalization, 5003
internal-test examples, or 7.52%, match full training pairs. Manual inspection
shows that many of these are caused by HTML escaping differences, casing
differences, repeated web templates, product fragments, navigation strings, and
subtitle-like markup.

Removing normalized parallel-pair overlaps lowers full internal BLEU from 60.5
to 59.2. This means near-duplicate contamination contributes somewhat to the
high internal score, but it does not fully explain it. Even after removing these
near-duplicate pairs, the internal score remains much higher than the official
score.

Source-only and target-only overlap is much larger, especially for the internal
test set, but these are weaker evidence of direct contamination than full
parallel-pair overlap. Many source-only or target-only matches are noisy,
template-like, or repeated web boilerplate. Removing these subsets does not
consistently lower the score, which suggests that they are not a simple
explanation for the high BLEU.

## Suggested Reporting Text

We checked for possible train/test contamination by comparing `data/full/train`
against both official and internal test sets. The official test set showed only
minor overlap: 13 normalized sentence-pair matches, corresponding to 1.06% of
the official test set. Removing these examples did not change the full model's
official score. The internal test set had no exact sentence-pair overlap, but it
did contain 5003 normalized pair overlaps, or 7.52%, mostly due to HTML
escaping, casing, and repeated web-template fragments. Removing these examples
reduced full internal BLEU from 60.5 to 59.2, indicating that near-duplicates
inflate the internal score somewhat but do not fully account for its high value.

## Detailed Files

Official test overlap details:

- `reports/data_preparation/official_test_train_overlap/contamination_check.md`
- `reports/data_preparation/official_test_train_overlap/summary.json`
- `reports/data_preparation/official_test_train_overlap/normalized_pair_matches.tsv`
- `reports/data_preparation/official_test_train_overlap/normalized_source_matches.tsv`
- `reports/data_preparation/official_test_train_overlap/normalized_target_matches.tsv`

Internal test overlap details:

- `reports/data_preparation/internal_test_train_overlap/contamination_check.md`
- `reports/data_preparation/internal_test_train_overlap/summary.json`
- `reports/data_preparation/internal_test_train_overlap/normalized_pair_matches.tsv`
- `reports/data_preparation/internal_test_train_overlap/normalized_source_matches.tsv`
- `reports/data_preparation/internal_test_train_overlap/normalized_target_matches.tsv`

