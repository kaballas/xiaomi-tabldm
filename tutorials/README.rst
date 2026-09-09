Tutorials
=========

A collection of runnable examples demonstrating typical TabLDM workflows.

The examples use ``cpu`` by default to make them portable. Set ``device="cuda"``
to run on a GPU.

Examples
--------

===============================  =====================================================
Example                          Demonstrates
===============================  =====================================================
``getting_started.py``           Basic classification, regression, and cross-validation
``mixed_data_types.py``          Numeric, categorical, boolean, and missing values
``horse_racing_top3.py``         Compare whole-race context sizes for top-three ranking
``classification_metrics.py``    Labels, probabilities, and classification metrics
``regression_quantiles.py``      Mean predictions and predictive quantiles
``kv_cache.py``                  Faster repeated prediction with one training context
``local_checkpoint.py``          Local checkpoint loading and CPU/CUDA selection
===============================  =====================================================

The horse-racing example reads its ordered model feature list from ``a.json``
at the repository root. Pass ``--features-json`` to use a different JSON file.
The repository-level ``finetune_tabldm.py`` script uses the same configuration
for experimental race-aware downstream weight training.

Checkpoint locations
--------------------

By default, TabLDM downloads the configured checkpoint from Hugging Face Hub on
the first ``fit()`` call. To run fully offline, pass an existing checkpoint file
to ``model_path``. The default Hub checkpoint paths are::

    checkpoints/clf_default.ckpt
    checkpoints/reg_default.ckpt

The examples also support local checkpoint overrides. When set, these
environment variables take precedence over the default Hub checkpoints::

    export TABLDM_CLF_CKPT=/path/to/classifier.ckpt
    export TABLDM_REG_CKPT=/path/to/regressor.ckpt
