# Data Quality in Haitian Creole-English Machine Translation

A group project for the **Machine Translation** course at **Uppsala University**.
We asked whether a smaller, cleaned parallel corpus could outperform a larger,
noisy corpus when training a Transformer to translate Haitian Creole into English.

## What we did

- Built a fairseq Transformer pipeline with a shared 8,000-token SentencePiece vocabulary.
- Compared full data, two cleaning strategies, and two size-matched random controls.
- Evaluated with sacreBLEU BLEU and chrF2, paired bootstrap tests, overlap checks, and manual error analysis.

## Results

These are the recorded final experiment results, evaluated on the original official test set of 1,228 pairs.

| Training condition | Training pairs | Official BLEU | Official chrF2 |
| --- | ---: | ---: | ---: |
| Full data | 532,027 | **43.0** | **64.8** |
| Light cleaning | 466,288 | 41.9 | 63.9 |
| Light random control | 466,288 | 42.4 | 64.4 |
| Risk-aware cleaning | 278,879 | 41.6 | 63.5 |
| Risk random control | 278,879 | 40.0 | 62.5 |

The full-data model performed best in this experiment. Filtering removed useful
coverage along with noise. This is a finding about this corpus and setup, rather
than a general claim that cleaning never helps.

## My contribution

This was a three-person group project. **Huanyu Dong** helped develop the research
question, built the main fairseq training framework, trained two experimental
models, analysed 50 translation examples, and wrote the corresponding report
sections. Model design, experiment planning, and interpretation were collaborative.
See [the personal contribution statement](CONTRIBUTIONS.md).

## Explore the project

- [Full report (LaTeX source)](reports/report.tex)
- [Final experiment results](results/converged_results.md)
- [Significance tests](results/significance_tests.md)
- [Cleaned-test robustness check](results/cleaned_test_evaluation.md)
- [Train/test overlap analysis](results/train_test_overlap.md)
- [Cleaning rules](docs/data_cleaning_annotation.md)
- [Training and data-preparation code](scripts/)

## Data

The experiments used the **Kreyol-MT** Haitian Creole-English corpus:
[upstream repository](https://github.com/JHU-CLSP/Kreyol-MT) and
[dataset](https://huggingface.co/datasets/jhu-clsp/kreyol-mt).
Obtain the data from its original source and follow its usage terms.
This repository contains our experiment code and summaries; it does not bundle
the corpus or trained checkpoints.

The original local export contained 665,033 training pairs and 1,228 official test
pairs. The training data was split 80/10/10 with seed 135, leaving 532,027 training
pairs, 66,503 development pairs, and 66,503 internal-test pairs. Upstream releases
may differ from that export, so record the exact data version for a new run.

Place aligned UTF-8 text files at:

```text
Kreyol-MT-main/eng-hat/train.hat
Kreyol-MT-main/eng-hat/train.eng
Kreyol-MT-main/eng-hat/test.hat
Kreyol-MT-main/eng-hat/test.eng
```

Explicit input paths can also be supplied to `scripts/prepare_mt_data.py`.

## Run the experiments

The environment files record the original Python 3.10/fairseq 0.12.3 setup.
They may require platform-specific dependency adjustments. Choose the CPU or GPU
file as appropriate; the GPU file records the CUDA 12.4 training environment.

```bash
conda env create -f env/gpu.yml
conda activate mt

python scripts/prepare_mt_data.py --seed 135
python scripts/create_matched_controls.py --light-seed 1135 --risk-seed 1136

RUN_ROOT="$PWD/experiments/fairseq_nmt_converged" \
MAX_UPDATE=200000 PATIENCE=25 \
bash scripts/run_all_fairseq_nmt_long.sh
```

The scripts use the active conda environment. Alternatively, set `MT_ENV` to an
existing environment's absolute path. For CPU debugging, use `env/cpu.yml` and
set `USE_CPU=1 USE_FP16=0`. Full model training requires substantial compute.

### Overlap and cleaned-test checks

```bash
python scripts/check_train_test_overlap.py \
  --train-src data/full/train.hat --train-tgt data/full/train.eng \
  --test-src data/official_test/test.hat --test-tgt data/official_test/test.eng \
  --output-dir reports/data_preparation/official_test_train_overlap

python scripts/clean_parallel_data.py \
  --src data/official_test/test.hat --tgt data/official_test/test.eng \
  --output-dir data/official_test_cleaned \
  --report-dir reports/data_preparation/official_test_cleaning

CONDITION=light_clean TEST_CONDITION=light_clean \
  bash scripts/run_cleaned_official_test_eval.sh
CONDITION=risk_clean TEST_CONDITION=risk_clean \
  bash scripts/run_cleaned_official_test_eval.sh
```

For significance tests, compare aligned hypotheses against the same references
using sacreBLEU paired bootstrap resampling (10,000 samples in the recorded study).

## Scope and limitations

This is a portfolio copy of the course project. The historical results are
preserved; the experiments were not retrained when this repository was prepared.
Shell scripts were adapted to use the active conda environment instead of a
personal machine path. Cleaning rules and model settings were not redesigned.

The manual analysis covered 150 examples across the team. We are not fluent
Haitian Creole speakers, so it focused on English outputs and references. Surface
filters and overlap metrics cannot fully measure translation quality. The legacy
training scripts allow skipping overlength validation/test inputs; when rerunning,
check that decoded hypotheses and references cover the same aligned examples.
