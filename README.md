<h3 align="center">
  <b>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
    Xiaomi-TabLDM: A Tabular Large Data Foundation Model<br/>For Classification and Regression via In-Context Learning
    <br/>
    <span>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>
    <br/>
  </b>
</h3>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/occams/Xiaomi-TabLDM" target="_blank">🤗 HuggingFace</a>
  &nbsp;|
  <a href="https://arxiv.org/abs/2609.03880" target="_blank">📔 Technical Report</a>
  &nbsp;|
  <a href="README_CN.md" target="_blank">中文</a>
  &nbsp;|
  English
  &nbsp;
</div>

<br/>

---

This repository is the official implementation of **Xiaomi-TabLDM**.

Tabular foundation models establish a general prediction paradigm based on in-context learning. Given labeled samples from a downstream dataset as context, a single pretrained model can make predictions directly without task-specific training. Building on this paradigm, we introduce Xiaomi-TabLDM, a tabular large data foundation model for classification and regression via in-context learning, which delivers superior prediction accuracy without requiring task-specific fine-tuning. Pretrained exclusively on synthetic data generated from structural causal models (SCMs), our model enables more flexible context utilization and more efficient capacity scaling.

**A New Performance Standard.** _Strong regression performance across benchmarks_: Xiaomi-TabLDM ranks 1st on OpenML-CTR23 and 2nd on regression across TALENT, TabArena, and BCCO, demonstrating consistently strong regression performance across four complementary benchmark suites. _Favorable performance–efficiency trade-off_: Xiaomi-TabLDM combines strong predictive performance with substantially lower computational cost. For example, on TabArena regression, it achieves the second-highest Elo while using 82% less training time and 68% less prediction time than the top-ranked TabFM.

**Large-Scale Synthetic Pretraining.** Xiaomi-TabLDM expands the coverage and diversity of synthetic tabular data used for pretraining. We also adopt a three-stage training strategy together with dual-stream feature grouping, lightweight Attention Residual, and sparse Mixture-of-Experts, enabling Xiaomi-TabLDM to learn richer feature interactions and expert specialization across diverse tabular tasks.

**Test-Time Scaling.** Xiaomi-TabLDM further extends tabular prediction through test-time compute scaling: allocating additional computation at inference time consistently improves predictive performance over the base model.

**Easy to Use:** Xiaomi-TabLDM can be installed with `pip` and provides a scikit-learn-compatible interface. `fit` does not update model weights; it only preprocesses the context and loads the pretrained model. Predictions are produced through in-context learning in a single forward pass.

**Fast:** With KV caching, repeated calls to `predict` on the same training data can reuse cached context projections, significantly accelerating repeated inference. A GPU is recommended for larger datasets, and CPU/disk offloading can be used to scale to larger data sizes.

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_Framework.png"
    alt="Xiaomi-TabLDM"
    width="800"
  >
</div>

## Performance

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TALENT.png"
    alt="Regression average-rank performance on TALENT (lower is better)"
    width="800"
  >
  <br>
  <em>Figure 1. Regression average-rank performance on TALENT (lower is better)</em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_TabArena.png"
    alt=" Regression Elo performance on TabArena"
    width="800"
  >
  <br>
  <em>Figure 2. Regression Elo performance on TabArena (higher is better).</em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_BCCO.png"
    alt="Performance on BCCO"
    width="800"
  >
  <br>
  <em>Figure 3. Average-rank comparison on BCCO. Circles denote the average ranks on BCCO-CLS and BCCO-REG, while diamonds denote the overall average rank across the two settings. Models are ordered
by the combined average rank; lower is better. </em>
</div>

<br>

<div align="center">
  <img
    src="assets/Xiaomi-TabLDM_OpenML-CTR23.png"
    alt="Performance on OpenML-CTR23"
    width="800"
  >
  <br>
  <em>Figure 4. Average-rank comparison on OpenML-CTR23 over 33 regression datasets (lower is better).</em>
</div>

## Installation

```bash
cd xiaomi-tabldm
pip install .
```

Install optional dependencies as needed:

```bash
pip install .[numba]   # Optional JIT acceleration for the quantile distribution layer
pip install .[test]    # Test dependencies
```

Installing PyTorch with `pip` may fail on Intel Macs. If so, install PyTorch first:

```bash
conda install pytorch -c pytorch
```

### Dependencies

`torch>=2.2`, `scikit-learn>=1.3.0`, `numpy`, `scipy`, `einops>=0.7`,
`psutil`, `tqdm>=4.64.0`, and `huggingface-hub`. `numba` is optional.

## Basic Usage

### Classification

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(model_path="checkpoints/clf_default.ckpt")
clf.fit(X_train, y_train)          # In-context learning: no weight updates
pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)  # (n_test, n_classes)
```

### Regression

```python
from tabldm import TabLDMRegressor

reg = TabLDMRegressor(model_path="checkpoints/reg_default.ckpt")
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
```

> `fit` **does not train the model**. It only preprocesses the labeled context
> (`X_train`, `y_train`) and loads the pretrained weights. Prediction is performed
> entirely through in-context learning. On first use, the checkpoint is downloaded
> automatically from the Hugging Face Hub. Specify `model_path` to use a local file
> for offline inference.

### KV Cache

When calling `predict` multiple times with the same training data, such as during
evaluation, enabling KV caching avoids repeatedly computing the context. The cache
is built during `fit` and reused across subsequent `predict` calls. Note that this
requires additional GPU/CPU memory, so choose the setting based on your use case:

> KV caching is not supported for classification tasks with more than 10 classes.
> Keep `kv_cache=False` (the default) for these datasets; otherwise `fit` raises an
> error.

```python
clf = TabLDMClassifier(
    kv_cache=True, model_path="checkpoints/clf_default.ckpt"
)
clf.fit(X_train, y_train)          # Build the cache once
clf.predict(X_test_batch_1)        # Reuse the cached context
clf.predict(X_test_batch_2)
```

### Save/Load

```python
clf.save(
    "classifier.pkl",
    save_model_weights=False,  # If False, reload weights from the checkpoint
    save_training_data=True,   # If True, include training data; False improves privacy
    save_kv_cache=True,        # Save the KV cache when available
)

from tabldm import TabLDMClassifier
clf = TabLDMClassifier.load("classifier.pkl")
```

When `save_model_weights=False` (the default), the saved file is smaller, but the
weights must be reloaded from `model_path` or the Hub when loading the estimator.

### Experimental Horse-Racing Training

The repository-level training scripts construct each episode from complete
earlier context races and one later query race. They use only the fields listed
in `a.json`; `race_id` controls chronological grouping and is not a model
feature.

#### Fine-tuning a pretrained checkpoint

`finetune_tabldm.py` provides downstream weight training using chronological
horse-racing splits from `Kaballas/races` by default.

```bash
# Fastest/safest first experiment: train only the prediction decoder.
python finetune_tabldm.py \
  --dataset Kaballas/races \
  --finetune-mode decoder \
  --context-races 10 \
  --epochs 20

# Small CPU smoke test without writing a checkpoint.
python finetune_tabldm.py \
  --dataset Kaballas/races \
  --device cpu \
  --context-races 1 \
  --epochs 1 \
  --max-train-races 1 \
  --max-validation-races 1 \
  --no-save
```

The available stages are `decoder`, `icl`, `row_icl`, and `full`. Complete
fine-tuned checkpoints remain compatible with `TabLDMClassifier(model_path=...)`.
Use the same feature JSON at inference time; `tutorials/horse_racing_top3.py`
checks the adjacent checkpoint metadata when available. Development runs do not
read the sealed test CSV. Add `--evaluate-test` only to the final selected run.

Checkpoint selection prioritizes validation Top-3 recall, then exact 3/3 race
rate, with race log loss as the tie-breaker. A controlled listwise-loss sweep
can be run by repeating the decoder experiment with `--listwise-weight` set to
`0`, `0.1`, `0.25`, and `0.5`.

```bash
python tutorials/horse_racing_top3.py \
  --checkpoint results/tabldm_horse_finetuned.ckpt \
  --features-json a.json \
  --context-races 10 \
  --n-estimators 1
```

#### Training a horse-racing model from scratch

`train_tabldm_from_scratch.py` constructs a new binary TabLDM classifier with
randomly initialized weights and trains every parameter. It does not read a
pretrained checkpoint. The resulting model is specific to this horse-racing
task; it does not reproduce Xiaomi's general foundation-model pretraining,
whose synthetic-data pipeline is not included in this inference-oriented
repository.

For example, the following command trains the smaller model architecture with
100 complete context races and up to 5,000 training episodes:

```bash
python train_tabldm_from_scratch.py \
  --device cpu \
  --epochs 5 \
  --warmup-epochs 0 \
  --context-races 100 \
  --max-train-races 5000 \
  --max-validation-races 2000 \
  --embed-dim 16 \
  --col-blocks 1 \
  --col-heads 2 \
  --col-inducing-points 8 \
  --row-blocks 1 \
  --row-heads 2 \
  --row-cls-tokens 2 \
  --icl-blocks 2 \
  --icl-heads 2 \
  --moe-layers all
```

The scratch trainer uses linear warmup followed by cosine learning-rate decay.
`--warmup-epochs 0` disables only warmup, not decay. With the defaults
`--learning-rate 3e-4`, `--min-learning-rate 3e-5`, and `--epochs 5`, the
epoch learning rates are approximately `3e-4`, `2.6e-4`, `1.65e-4`,
`6.95e-5`, and `3e-5`. To use a constant learning rate, set the minimum equal
to the initial rate:

```bash
python train_tabldm_from_scratch.py \
  --dataset Kaballas/races \
  --preprocessing-workers 4 \
  --batch-size 8 \
  --learning-rate 3e-4 \
  --min-learning-rate 3e-4 \
  --warmup-epochs 0
```

The scratch trainer downloads `training.csv` and `validation.csv` from the
Hugging Face dataset cache. `test.csv` is downloaded only when
`--evaluate-test` is passed. To use local data instead, pass both `--train-csv`
and `--validation-csv` (and `--test-csv` when evaluating the test split).
Its default model arguments reproduce the released classifier architecture:
128-dimensional column embeddings, 3 column blocks, 3 row blocks with 4 CLS
tokens and RoPE, 24 ICL blocks with the final 8 using 2-expert top-1 MoE, and a
10-class target encoder/decoder. The horse task loss uses active classes 0 and
1. MoE load-balance and router z-loss terms are summed over the MoE layers and
use the paper's default coefficients and global auxiliary weight.
`--batch-size` stacks episodes with matching context, query, and feature
dimensions; smaller incompatible groups are processed as partial batches.
Episode preparation uses up to four CPU processes by default; tune this with
`--preprocessing-workers`, or pass `1` for serial preprocessing.

The default output is
`results/tabldm_horse_from_scratch.ckpt`, accompanied by a JSON metadata file.

#### Predicting with the scratch-trained model

`predict_scratch_model.py` loads the scratch checkpoint, fits the labelled
in-context races, scores each prediction race independently, and selects the
three runners with the highest class-1 probabilities. Use the same context size
used for training:

```bash
python predict_scratch_model.py \
  --checkpoint results/tabldm_horse_from_scratch.ckpt \
  --context-races 100 \
  --kv-cache kv \
  --device cpu
```

By default it uses `data/test.csv` as labelled context, reads runners from
`data/predict.csv`, and writes `results/tabldm_predictions.csv`. Pre-result
rows are all scored even when `runner_mask` is zero. Pass `--use-runner-mask`
only when the prediction file uses `runner_mask=1` to identify active runners.
The default `--kv-cache kv` builds the context cache once during `fit` and
reuses it for every compatible prediction race. Use `--kv-cache repr` to trade
some speed for substantially lower cache memory, or `--kv-cache off` to disable
it. A race with a configured feature that is entirely missing requires a
different feature layout; the script reports that race and safely bypasses the
cache for it.

#### Fine-tuning the scratch-trained model

The saved scratch checkpoint is compatible with `finetune_tabldm.py`. A safe
first experiment updates only its decoder while preserving the representations
learned during scratch training:

```bash
python finetune_tabldm.py \
  --dataset Kaballas/races \
  --checkpoint results/tabldm_horse_from_scratch.ckpt \
  --output results/tabldm_horse_scratch_finetuned.ckpt \
  --device cpu \
  --finetune-mode decoder \
  --context-races 100 \
  --epochs 20 \
  --learning-rate 1e-4 \
  --listwise-weight 0.1 \
  --gradient-accumulation 4
```

To update every parameter, use a smaller learning rate:

```bash
python finetune_tabldm.py \
  --dataset Kaballas/races \
  --checkpoint results/tabldm_horse_from_scratch.ckpt \
  --output results/tabldm_horse_scratch_full_finetuned.ckpt \
  --device cpu \
  --finetune-mode full \
  --context-races 100 \
  --epochs 10 \
  --learning-rate 1e-5 \
  --listwise-weight 0.1 \
  --gradient-accumulation 4
```

Unlike the scratch trainer, `finetune_tabldm.py` uses a constant learning rate.
Keep the original scratch checkpoint until the fine-tuned checkpoint has been
compared on validation data. To predict with the selected fine-tuned decoder:

```bash
python predict_scratch_model.py \
  --checkpoint results/tabldm_horse_scratch_finetuned.ckpt \
  --context-races 100 \
  --kv-cache kv \
  --device cpu
```

## Advanced Configuration

Xiaomi-TabLDM provides a set of parameters for customizing inference behavior. The following
example shows all available classifier parameters and their default values:

```python
from tabldm import TabLDMClassifier

clf = TabLDMClassifier(
    n_estimators=8,               # Ensemble members; more is more accurate but slower
    norm_methods=None,            # Normalization methods to try
    feat_shuffle_method="latin",  # Feature permutation strategy
    class_shuffle_method="shift", # Class permutation strategy
    outlier_threshold=4.0,        # Z-score threshold for outlier detection/clipping
    softmax_temperature=0.9,      # Temperature controlling prediction confidence
    average_logits=True,          # Average logits (True) or probabilities (False)
    support_many_classes=True,    # Automatically handle more than 10 classes
    batch_size=4,                 # Ensemble members processed together; lower saves memory
    kv_cache=False,               # Cache training-data KV projections for repeated inference
    model_path=None,              # Checkpoint path; None downloads from Hugging Face
    allow_auto_download=True,     # Download automatically when not found locally
    checkpoint_version="checkpoints/clf_default.ckpt",  # Pretrained checkpoint version
    device=None,                  # Inference device; None selects CUDA or CPU automatically
    use_amp="auto",               # Automatic mixed precision for faster inference
    use_fa3="auto",               # Flash Attention 3 on Hopper GPUs such as H100
    offload_mode="auto",          # Decide automatically when to use CPU/disk offloading
    disk_offload_dir=None,        # Directory for disk offloading
    random_state=42,              # Random seed for reproducibility
    n_jobs=None,                  # Number of PyTorch threads for CPU inference
    verbose=False,                # Print detailed inference information
    inference_config=None,        # Fine-grained inference control for advanced users
)
```

`TabLDMRegressor` accepts the same parameters except for the classification-specific
parameters `class_shuffle_method`, `softmax_temperature`, `average_logits`, and
`support_many_classes`.



## Loading Checkpoints

Checkpoints are resolved in the following order:

1. **`model_path`** — If it points to an existing file, that file is used directly.
2. If `model_path` is set but the file does not exist and `allow_auto_download=True`,
   the checkpoint named by `checkpoint_version` is downloaded to `model_path`.
3. If `model_path` is `None`, the checkpoint is retrieved from the Hugging Face Hub
   cache using `checkpoint_version` as the key.


The `checkpoint_version` value is the filename inside the Hugging Face repository, not a
local filesystem path. The first lookup uses the local Hugging Face cache
(typically `~/.cache/huggingface/hub`); if the file is not cached and
`allow_auto_download=True`, it is downloaded automatically.

## Available Models

| Model          | Classifier           | Regressor           |
| -------------- | -------------------- | ------------------- |
| **Xiaomi-TabLDM** | [`XiaomiTabLDMClassifier`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/clf_default.ckpt) | [`XiaomiTabLDMRegressor`](https://huggingface.co/occams/Xiaomi-TabLDM/resolve/main/checkpoints/reg_default.ckpt) |


## Testing

```bash
cd xiaomi-tabldm
pytest tests/test_infer_package.py -v
```

By default, the tests look for checkpoints in `../checkpoints`. Override
this location with the `TABLDM_CKPT_DIR` environment variable. If no checkpoint is
found, the tests are skipped automatically.

## License

This project is released under the [Apache License 2.0](LICENSE).

Copyright (C) 2026 Xiaomi Corporation

## FAQ

**What is Xiaomi-TabLDM?**
Xiaomi-TabLDM is a tabular foundation model similar to TabPFN and TabICL. It learns new data
through in-context learning in a single forward pass of a pretrained Transformer:
`y_pred = model(X_train, y_train, X_test)` (called internally by `predict()`).
Its learning capability comes from pretraining on large-scale synthetic data.

**How fast is Xiaomi-TabLDM?**
For a dataset with $n$ training rows and $m$ columns, the runtime complexity is
$O(n^2 + nm^2)$. KV caching accelerates repeated inference on the same training data,
while CPU/disk offloading enables larger datasets to be processed without running
out of memory.

**What dataset sizes are suitable?**
The pretraining data covers hundreds to tens of thousands of training samples and
datasets ranging from a few to more than one hundred feature columns. The model can
extrapolate to larger scales, although accuracy may decline as the data moves beyond
the training distribution. Specific recommended ranges will be added after empirical
evaluation.

## Preprocessing

### Built-In Preprocessing

For `X`, Xiaomi-TabLDM accepts either a pandas DataFrame or a NumPy array and performs the
following operations:

- Detect and ordinal-encode categorical columns, including string, object, category,
  and boolean columns. In NumPy arrays, all columns share the same data type, and
  integer columns are treated as numerical.
- Create a separate category for missing values in categorical features.
- Mean-impute missing numerical values encoded as NaN.
- Detect and clip outliers.
- Scale and normalize features.
- Permute features to increase ensemble diversity.

## Package Layout

```
tabldm/
├── __init__.py          # Public API: estimators + InferenceConfig
├── __about__.py         # Version number
├── _model/              # PyTorch model + inference engine
│   ├── tabldm.py                 # Base TabLDM module
│   ├── attnres_light_rmsnorm.py # AttnRes/RMSNorm architecture
│   ├── attnres_light_rmsnorm_moe.py # MoE architecture
│   ├── embedding*.py, interaction.py, learning.py, encoders.py, layers.py
│   ├── attention.py, rope.py, ssmax.py, moe.py, quantile_dist.py
│   ├── kv_cache.py, kv_cache_attnres.py
│   ├── inference.py, inference_config.py
└── _sklearn/            # scikit-learn interface
    ├── base.py, classifier.py, regressor.py
    ├── preprocessing.py, sklearn_utils.py
    └── *_dualstream_moe.py   # MoE estimators
```

## Citation
```bibtex
@misc{tabldmteam2026xiaomitabldmtabularfoundationmodel,
  title         = {Xiaomi-TabLDM: A Tabular Foundation Model Technical Report},
  author        = {Penghui Wang and Wei Liu and Hong Wang and Chengyue Huang and Yuxi Sun and Zirui Wang and Hongming Huang and Quan Wang and Zhenwei Xin and Ping Hou and Jie Yu and Chunxiao Liu and Erli Meng and Bin Wang},
  year          = {2026},
  eprint        = {2609.03880},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.03880},
}
```
