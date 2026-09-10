from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tabldm._model.attnres_light_rmsnorm_moe import (  # noqa: E402
    AttnResEncoderLightRMSNormMoE,
    TabLDMSparseMoE,
)
from tabldm._model.moe import (  # noqa: E402
    SparseMoEFeedForward,
    collect_moe_aux_loss,
)
from train_tabldm_from_scratch import build_model_config, parse_args  # noqa: E402


def test_scratch_defaults_match_released_classifier_architecture(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_tabldm_from_scratch.py"])
    args = parse_args()
    config = build_model_config(args)

    expected = {
        "max_classes": 10,
        "embed_dim": 128,
        "col_feature_group_size": 3,
        "global_dilation": "adaptive",
        "global_max_span": 32,
        "col_num_blocks": 3,
        "col_nhead": 8,
        "col_num_inds": 128,
        "row_num_blocks": 3,
        "row_nhead": 8,
        "row_num_cls": 4,
        "row_rope_base": 100000,
        "icl_num_blocks": 24,
        "icl_nhead": 8,
        "ff_factor": 2,
        "block_size": 4,
        "attnres_stride": 4,
        "moe_layers": "last_8",
        "moe_num_experts": 2,
        "moe_top_k": 1,
        "moe_num_shared_experts": 1,
        "moe_router_z_loss_coef": 1e-3,
        "moe_load_balance_loss_coef": 1e-2,
        "moe_router_jitter": 0.0,
        "moe_init_from_dense": True,
    }
    assert {key: config[key] for key in expected} == expected
    assert config["embed_dim"] * config["row_num_cls"] == 512
    assert config["embed_dim"] * config["row_num_cls"] * config["ff_factor"] == 1024
    assert args.moe_aux_weight == 1.0
    assert AttnResEncoderLightRMSNormMoE._resolve_moe_layers(24, "last_8") == set(
        range(16, 24)
    )


def test_moe_auxiliary_losses_are_summed_across_layers():
    layers = torch.nn.ModuleList(
        [
            SparseMoEFeedForward(4, 8, num_experts=2, top_k=1),
            SparseMoEFeedForward(4, 8, num_experts=2, top_k=1),
        ]
    )
    layers[0]._last_aux = {"z_loss": torch.tensor(1.25)}
    layers[1]._last_aux = {"load_balance_loss": torch.tensor(2.75)}

    assert torch.equal(collect_moe_aux_loss(layers), torch.tensor(4.0))


def test_reduced_model_preserves_released_architecture_wiring(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_tabldm_from_scratch.py"])
    args = parse_args()
    args.embed_dim = 16
    args.col_blocks = 1
    args.col_heads = 2
    args.col_inducing_points = 8
    args.row_blocks = 2
    args.row_heads = 2
    args.row_cls_tokens = 2
    args.icl_blocks = 9
    args.icl_heads = 2

    model = TabLDMSparseMoE(**build_model_config(args))
    icl = model.icl_predictor

    assert model.row_interactor.encoder_prefix.rope is not None
    assert model.row_interactor.encoder_prefix.block_size == 4
    assert model.row_interactor.encoder_prefix.attnres_stride == 4
    assert icl.y_encoder.in_features == 10
    assert icl.y_encoder.out_features == 32
    assert (icl.decoder[0].in_features, icl.decoder[0].out_features) == (32, 64)
    assert (icl.decoder[2].in_features, icl.decoder[2].out_features) == (64, 10)

    moe_layers = [
        layer.moe_ffn for layer in icl.tf_icl.layers if hasattr(layer, "moe_ffn")
    ]
    assert len(moe_layers) == 8
    assert all(len(layer.experts) == 2 for layer in moe_layers)
    assert all(layer.top_k == 1 for layer in moe_layers)
    assert all(len(layer.shared_experts) == 1 for layer in moe_layers)
    assert all(layer.experts[0].linear1.out_features == 64 for layer in moe_layers)
    assert all(layer.router_z_loss_coef == 1e-3 for layer in moe_layers)
    assert all(layer.load_balance_loss_coef == 1e-2 for layer in moe_layers)
    assert all(layer.router_jitter == 0.0 for layer in moe_layers)
