"""Checks that the Boolean projection preserves charge and circuit semantics."""

import unittest

import torch
from torchlogix import Circuit

from .model import CircuitReadout, LogicRegressor, charge_sum_projections


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


if __name__ == "__main__":
    unittest.main()
