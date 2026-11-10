"""Validate the paper-style Conv1D charge projection and compact RTL boundary."""

import copy
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torchlogix import Circuit

from .export import count_rtl
from .model import CompactChargeCircuitReadout, LogicRegressor, charge_sum_projections


class CompactConv1DTest(unittest.TestCase):
    def test_projection_keeps_axis_order_and_exact_charge(self):
        codes = torch.zeros((1, 2, 16, 16), dtype=torch.uint8)
        codes[0, 0, 4, 9] = 3
        codes[0, 1, 7, 2] = 2
        bits = (codes.unsqueeze(2) >= torch.arange(1, 4).view(1, 1, 3, 1, 1)).flatten(1, 2)
        x, y = charge_sum_projections(bits, 2)
        powers = (2 ** torch.arange(6)).view(1, 1, 1, 6)
        x_values = (x.reshape(1, 2, 16, 6).long() * powers).sum(-1)
        y_values = (y.reshape(1, 2, 16, 6).long() * powers).sum(-1)
        torch.testing.assert_close(x_values, codes.sum(3))
        torch.testing.assert_close(y_values, codes.sum(2))
        self.assertEqual(int(x_values[0, 0, 4]), 3)
        self.assertEqual(int(y_values[0, 0, 9]), 3)
        self.assertEqual(int(x_values[0, 1, 7]), 2)
        self.assertEqual(int(y_values[0, 1, 2]), 2)

    def test_compact_circuit_and_rtl_counts(self):
        torch.manual_seed(27)
        readout_bits = 8
        model = LogicRegressor("conv1d", "slim", 2, readout_bits, "cpu",
                               projection="charge_sum").eval()
        core = CompactChargeCircuitReadout(model)
        codes = torch.randint(0, 4, (3, 2, 16, 16))
        bits = (codes.unsqueeze(2) >= torch.arange(1, 4).view(1, 1, 3, 1, 1)).flatten(1, 2)
        x, y = charge_sum_projections(bits, 2)
        projected = torch.cat((x, y), 1)
        self.assertEqual(projected.shape, (3, 384))
        torch.testing.assert_close(model.forward_logic(bits), core(projected))

        circuit = Circuit.from_model(core, input_shape=(384,))
        self.assertEqual(len(circuit.sum_nodes), 3)
        torch.testing.assert_close(core(projected), circuit(projected), atol=1e-6, rtol=0)
        reductions = {sr.node_id: sr for sr in circuit.sum_nodes}
        gates = copy.deepcopy(circuit)
        gates.outputs = [wire for output in circuit.outputs
                         for wire in reductions[output].input_ids]
        gates.sum_nodes = []
        gates.output_shape = [3 * readout_bits]
        gate_output = gates(projected).reshape(3, 3, readout_bits)
        counts = gate_output.long().sum(2)
        expected = torch.cat([
            (counts[:, i:i + 1] + readout.beta) / readout.tau
            for i, readout in enumerate(model.readouts)
        ], 1)
        torch.testing.assert_close(core(projected), expected, atol=1e-6, rtol=0)

        if not shutil.which("iverilog") or not shutil.which("vvp"):
            return
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            gates.simplify()
            gates.write_verilog_code(str(folder / "gates.v"))
            (folder / "compact_top.sv").write_text(count_rtl(readout_bits, 384))
            count_bits = math.ceil(math.log2(readout_bits + 1))
            vectors = []
            for row, count in zip(projected, counts):
                inp = sum(int(bit) << i for i, bit in enumerate(row))
                out = sum(int(value) << (i * count_bits) for i, value in enumerate(count))
                vectors.append(f"    projected_bits = 384'h{inp:x}; #1; "
                               f"if (counts_flat !== {3 * count_bits}'h{out:x}) $fatal(1, \"count mismatch\");")
            (folder / "tb.sv").write_text(
                "module tb;\n"
                "  reg [383:0] projected_bits;\n"
                f"  wire [{3 * count_bits - 1}:0] counts_flat;\n"
                "  lnn_conv1d_slim_compact dut (.projected_bits(projected_bits), .counts_flat(counts_flat));\n"
                "  initial begin\n" + "\n".join(vectors) + "\n    $finish;\n  end\nendmodule\n")
            subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(folder / "sim"),
                            str(folder / "gates.v"), str(folder / "compact_top.sv"),
                            str(folder / "tb.sv")], check=True, capture_output=True, text=True)
            subprocess.run(["vvp", str(folder / "sim")], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
