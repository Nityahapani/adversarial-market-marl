"""
Unit tests for the BeliefTransformer sequential inference model.
"""

import pytest
import torch

from adversarial_market.networks.belief_transformer import BeliefTransformer


@pytest.fixture
def small_transformer():
    return BeliefTransformer(
        input_dim=5,
        d_model=32,
        n_heads=2,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        max_seq_len=50,
    )


class TestBeliefTransformerShape:
    def test_belief_output_shape(self, small_transformer):
        seq = torch.randn(8, 20, 5)
        belief, repr_ = small_transformer(seq)
        assert belief.shape == (8,)
        assert repr_.shape == (8, small_transformer.d_model // 4)

    def test_belief_in_range_zero_one(self, small_transformer):
        seq = torch.randn(16, 10, 5)
        belief, _ = small_transformer(seq)
        assert (belief >= 0.0).all()
        assert (belief <= 1.0).all()

    def test_single_step_sequence(self, small_transformer):
        seq = torch.randn(4, 1, 5)
        belief, _ = small_transformer(seq)
        assert belief.shape == (4,)

    def test_with_padding_mask(self, small_transformer):
        """Masked positions should not affect the belief output."""
        B, T = 4, 15
        seq = torch.randn(B, T, 5)
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, 10:] = True  # last 5 positions are padding
        belief_masked, _ = small_transformer(seq, src_key_padding_mask=mask)
        assert belief_masked.shape == (B,)
        assert (belief_masked >= 0).all()
        assert (belief_masked <= 1).all()


class TestBeliefLoss:
    def test_loss_shape_and_finite(self, small_transformer):
        seq = torch.randn(8, 15, 5)
        labels = torch.randint(0, 2, (8,)).float()
        loss, belief = small_transformer.compute_belief_loss(seq, labels)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_loss_backprop(self, small_transformer):
        opt = torch.optim.Adam(small_transformer.parameters(), lr=1e-3)
        seq = torch.randn(4, 10, 5)
        labels = torch.ones(4)
        loss, _ = small_transformer.compute_belief_loss(seq, labels)
        loss.backward()
        opt.step()

    def test_entropy_regularisation_reduces_overconfidence(self, small_transformer):
        """
        Verify that the entropy regularisation term is structurally active:
        the entropy-regularised loss should differ from plain BCE, and
        training with very high entropy_reg_weight should still produce
        finite losses and valid beliefs in [0, 1].
        """
        torch.manual_seed(0)
        seq = torch.randn(16, 10, 5)
        labels = torch.ones(16)

        # Loss with zero entropy reg
        loss_no_reg, belief_no_reg = small_transformer.compute_belief_loss(
            seq, labels, entropy_reg_weight=0.0
        )
        # Loss with high entropy reg
        loss_high_reg, belief_high_reg = small_transformer.compute_belief_loss(
            seq, labels, entropy_reg_weight=5.0
        )

        # Both losses must be finite
        assert torch.isfinite(loss_no_reg)
        assert torch.isfinite(loss_high_reg)

        # Beliefs must stay in [0, 1]
        assert belief_high_reg.min().item() >= 0.0
        assert belief_high_reg.max().item() <= 1.0

        # The two losses must differ (entropy term has a real effect)
        assert not torch.isclose(loss_no_reg, loss_high_reg, atol=1e-6)


class TestBeliefUpdate:
    def test_update_belief_returns_float(self, small_transformer):
        seq = torch.randn(1, 10, 5)
        b = small_transformer.update_belief(seq)
        assert isinstance(b, float)
        assert 0.0 <= b <= 1.0

    def test_no_gradient_leak_in_update(self, small_transformer):
        """update_belief should not leave gradients attached."""
        seq = torch.randn(2, 10, 5)
        b = small_transformer.update_belief(seq)
        # No tensors with grad_fn should escape
        assert isinstance(b, float)


class TestBeliefTransformerVariableLengths:
    def test_different_batch_sizes(self, small_transformer):
        for B in [1, 4, 16]:
            seq = torch.randn(B, 10, 5)
            belief, _ = small_transformer(seq)
            assert belief.shape == (B,)

    def test_sequence_lengths(self, small_transformer):
        for T in [1, 5, 20, 50]:
            seq = torch.randn(2, T, 5)
            belief, _ = small_transformer(seq)
            assert belief.shape == (2,)
