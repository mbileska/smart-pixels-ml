"""Checks that the Boolean projection preserves charge and circuit semantics."""

import unittest

import torch
from torchlogix import Circuit

from .model import (CHARGE_THRESHOLDS, CircuitReadout, LogicRegressor,
                    charge_sum_projections, or_charge_threshold_projections)


class ChargeProjectionTest(unittest.TestCase):
    def test_matches_average_charge_information(self):
        torch.manual_seed(7)
        codes = torch.cat((torch.zeros(1, 2, 16, 16, dtype=torch.uint8),
                           torch.full((1, 2, 16, 16), 3, dtype=torch.uint8),
                           torch.randint(0, 4, (16, 2, 16, 16), dtype=torch.uint8)))
        comparator_bits = (codes.unsqueeze(2) >=
                           torch.arange(1, 4).view(1, 1, 3, 1, 1)).flatten(1, 2)
        x, y = charge_sum_projections(comparator_bits, frames=2)
        weights = (2 ** torch.arange(6)).view(1, 1, 1, 6)
        decoded_x = (x.reshape(-1, 2, 16, 6).long() * weights).sum(-1)
        decoded_y = (y.reshape(-1, 2, 16, 6).long() * weights).sum(-1)
        torch.testing.assert_close(decoded_x, codes.sum(dim=3))
        torch.testing.assert_close(decoded_y, codes.sum(dim=2))

    def test_circuit_matches_pytorch(self):
        torch.manual_seed(11)
        model = LogicRegressor("mlp", "slim", 2, 8, "cpu", projection="charge_sum").eval()
        core = CircuitReadout(model)
        circuit = Circuit.from_model(core, input_shape=(6, 16, 16))
        self.assertEqual(len(circuit.sum_nodes), 3)
        bits = torch.randint(0, 2, (2, 6, 16, 16), dtype=torch.bool)
        torch.testing.assert_close(core(bits), circuit(bits), atol=1e-6, rtol=0)

    def test_or_charge_thresholds_are_exact_and_ordered(self):
        torch.manual_seed(17)
        codes = torch.cat((torch.zeros(1, 2, 16, 16, dtype=torch.uint8),
                           torch.full((1, 2, 16, 16), 3, dtype=torch.uint8),
                           torch.randint(0, 4, (16, 2, 16, 16), dtype=torch.uint8)))
        bits = (codes.unsqueeze(2) >= torch.arange(1, 4).view(1, 1, 3, 1, 1)).flatten(1, 2)
        x, y = or_charge_threshold_projections(bits, frames=2)
        self.assertEqual(x.shape, (len(codes), 256))
        for projected, axis in ((x, 3), (y, 2)):
            actual = projected.reshape(-1, 2, 8, 16)
            expected_or = torch.stack([(codes >= t).any(dim=axis) for t in (1, 2, 3)], dim=2)
            summed = codes.sum(dim=axis)
            expected_thresholds = torch.stack([summed >= t for t in CHARGE_THRESHOLDS], dim=2)
            torch.testing.assert_close(actual[:, :, :3], expected_or)
            torch.testing.assert_close(actual[:, :, 3:], expected_thresholds)
            self.assertTrue((actual[:, :, 3:-1] >= actual[:, :, 4:]).all())

    def test_or_charge_threshold_circuit_matches_pytorch(self):
        torch.manual_seed(19)
        model = LogicRegressor("mlp", "slim", 2, 8, "cpu",
                               projection="or_charge_thresholds").eval()
        core = CircuitReadout(model)
        circuit = Circuit.from_model(core, input_shape=(6, 16, 16))
        self.assertEqual(len(circuit.sum_nodes), 3)
        codes = torch.randint(0, 4, (2, 2, 16, 16), dtype=torch.uint8)
        bits = (codes.unsqueeze(2) >= torch.arange(1, 4).view(1, 1, 3, 1, 1)).flatten(1, 2)
        torch.testing.assert_close(core(bits), circuit(bits), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
