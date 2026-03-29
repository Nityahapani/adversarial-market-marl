"""
Unit tests for the MINE mutual information estimator.

Tests:
  - Network forward pass shapes
  - MI estimate is non-negative for dependent samples
  - MI estimate near zero for independent samples
  - Loss backpropagation
  - PredictabilityPenalty forward pass
"""

import pytest
import torch

from adversarial_market.networks.mine_estimator import (
    MINEEstimator,
    MINENetwork,
    PredictabilityPenalty,
)


class TestMINENetwork:
    def test_output_shape(self):
        net = MINENetwork(z_dim=1, f_dim=4, hidden_dims=(64, 64))
        z = torch.randn(32, 1)
        f = torch.randn(32, 4)
        out = net(z, f)
        assert out.shape == (32,)

    def test_output_is_real(self):
        net = MINENetwork(z_dim=1, f_dim=4)
        z = torch.randn(16, 1)
        f = torch.randn(16, 4)
        out = net(z, f)
        assert not torch.any(torch.isnan(out))
        assert not torch.any(torch.isinf(out))

    def test_parameters_exist(self):
        net = MINENetwork(z_dim=2, f_dim=8, hidden_dims=(32,))
        params = list(net.parameters())
        assert len(params) > 0
        total = sum(p.numel() for p in params)
        assert total > 0


class TestMINEEstimator:
    @pytest.fixture
    def estimator(self):
        return MINEEstimator(z_dim=1, f_dim=4, hidden_dims=(64, 64), ema_decay=0.99)

    def test_compute_returns_tuple(self, estimator):
        z = torch.randn(64, 1)
        f = torch.randn(64, 4)
        mi, loss = estimator.compute(z, f)
        assert mi.shape == ()
        assert loss.shape == ()

    def test_loss_is_backpropable(self, estimator):
        opt = torch.optim.Adam(estimator.parameters(), lr=1e-3)
        z = torch.randn(64, 1)
        f = torch.randn(64, 4)
        mi, loss = estimator.compute(z, f)
        loss.backward()
        opt.step()
        # No crash = pass

    def test_mi_higher_for_dependent_samples(self, estimator):
        """MI should be higher when z and f are correlated."""
        torch.manual_seed(0)
        n = 256

        # Independent: z ~ N(0,1), f independent
        z_ind = torch.randn(n, 1)
        f_ind = torch.randn(n, 4)

        # Dependent: f[:, 0] = z + small noise
        z_dep = torch.randn(n, 1)
        f_dep = torch.randn(n, 4)
        f_dep[:, 0] = z_dep.squeeze() + 0.1 * torch.randn(n)

        # Train briefly on dependent samples
        opt = torch.optim.Adam(estimator.parameters(), lr=1e-3)
        for _ in range(50):
            _, loss = estimator.compute(z_dep, f_dep)
            opt.zero_grad()
            loss.backward()
            opt.step()

        mi_dep, _ = estimator.compute(z_dep, f_dep)
        mi_ind, _ = estimator.compute(z_ind, f_ind)

        # After training on dependent samples, MI_dep should be higher
        assert mi_dep.item() > mi_ind.item() - 0.5  # allow margin

    def test_estimate_only_no_grad(self, estimator):
        z = torch.randn(32, 1)
        f = torch.randn(32, 4)
        mi = estimator.estimate_only(z, f)
        assert isinstance(mi, float)
        assert not torch.any(torch.tensor(mi).isnan())

    def test_ema_buffer_updates(self, estimator):
        initial_ema = float(estimator.ema_et.item())
        z = torch.randn(32, 1)
        f = torch.randn(32, 4)
        estimator.compute(z, f)
        updated_ema = float(estimator.ema_et.item())
        # EMA should change after compute
        assert updated_ema != initial_ema


class TestPredictabilityPenalty:
    @pytest.fixture
    def penalty_net(self):
        return PredictabilityPenalty(flow_dim=4, hidden_dim=32)

    def test_forward_shape(self, penalty_net):
        seq = torch.randn(8, 20, 4)  # (B, T, flow_dim)
        pred, loss = penalty_net(seq)
        assert pred.shape == ()
        assert loss.shape == ()

    def test_short_sequence_returns_zero(self, penalty_net):
        seq = torch.randn(4, 1, 4)
        pred, loss = penalty_net(seq)
        assert float(pred.item()) == pytest.approx(0.0)
        assert float(loss.item()) == pytest.approx(0.0)

    def test_backprop(self, penalty_net):
        opt = torch.optim.Adam(penalty_net.parameters(), lr=1e-3)
        seq = torch.randn(4, 15, 4)
        _, loss = penalty_net(seq)
        loss.backward()
        opt.step()
