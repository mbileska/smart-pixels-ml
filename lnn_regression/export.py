"""Export the binary LNN core with one calibrated sum reduction per feature."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from .model import (CircuitReadout, CompactChargeCircuitReadout, LogicRegressor,
                    charge_sum_projections)


def count_rtl(readout_bits, input_bits):
    """Combinational wrapper around the compact gate circuit's three readouts."""
    count_bits = math.ceil(math.log2(readout_bits + 1))
    lines = ["// Gate order: x, y, cotB; each group has readout_bits Boolean outputs.",
             "// Physical units: center[i] + scale[i] * (count[i] + beta[i]) / tau[i].",
             "module lnn_conv1d_slim_compact (",
             f"    input wire [{input_bits - 1}:0] projected_bits,",
             f"    output wire [{3 * count_bits - 1}:0] counts_flat",
             ");",
             f"    wire [{3 * readout_bits - 1}:0] gates;",
             "    circuit u_circuit (.inp(projected_bits), .out(gates));",
             f"    reg [{count_bits - 1}:0] count_x, count_y, count_cotB;",
             "    integer i;",
             "    always @* begin",
             "        count_x = 0; count_y = 0; count_cotB = 0;",
             f"        for (i = 0; i < {readout_bits}; i = i + 1) begin",
             "            count_x = count_x + gates[i];",
             f"            count_y = count_y + gates[i + {readout_bits}];",
             f"            count_cotB = count_cotB + gates[i + {2 * readout_bits}];",
             "        end",
             "    end",
             f"    assign counts_flat[{count_bits - 1}:0] = count_x;",
             f"    assign counts_flat[{2 * count_bits - 1}:{count_bits}] = count_y;",
             f"    assign counts_flat[{3 * count_bits - 1}:{2 * count_bits}] = count_cotB;",
             "endmodule"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from torchlogix import Circuit

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibration = checkpoint.get("calibration")
    model = LogicRegressor(checkpoint["architecture"], checkpoint["task"],
                           checkpoint["frames"], checkpoint["readout_bits"], "cpu",
                           calibration["readout_bounds"] if calibration else None,
                           checkpoint.get("projection", "or"))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    compact = checkpoint["architecture"] == "conv1d" and checkpoint.get("projection") == "charge_sum"
    core = CompactChargeCircuitReadout(model) if compact else CircuitReadout(model)
    input_shape = ((2 * checkpoint["frames"] * 16 * 6,) if compact else
                   (3 * checkpoint["frames"], 16, 16))
    circuit = Circuit.from_model(core, input_shape=input_shape)
    reductions = {sr.node_id: sr for sr in circuit.sum_nodes}
    output_reductions = [reductions.get(output) for output in circuit.outputs]
    if len(output_reductions) != len(model.readouts) or any(sr is None for sr in output_reductions):
        raise RuntimeError("Expected exactly one Circuit sum reduction per output feature")
    for i, (sr, group) in enumerate(zip(output_reductions, model.readouts)):
        if sr.tau != group.tau or abs(sr.beta - group.beta) > 1e-6:
            raise RuntimeError(f"Feature {i} lost GroupSum tau/beta during export")
    original_readouts = [{"tau": group.tau, "beta": group.beta} for group in model.readouts]
    if compact:
        if checkpoint["task"] != "slim" or len(output_reductions) != 3 or any(
                len(sr.input_ids) != checkpoint["readout_bits"] for sr in output_reductions):
            raise RuntimeError("Compact count RTL expects three complete Slim readout groups")
        gate_circuit = copy.deepcopy(circuit)
        gate_circuit.outputs = [wire for sr in output_reductions for wire in sr.input_ids]
        gate_circuit.sum_nodes = []
        gate_circuit.output_shape = [len(gate_circuit.outputs)]
        if len(gate_circuit.outputs) != 3 * checkpoint["readout_bits"]:
            raise RuntimeError("Compact gate circuit has the wrong output width")
        # Check the integer count boundary before the calibrated tau/beta step.
        for _ in range(2):
            projected_bits = torch.randint(0, 2, (1, input_shape[0]), dtype=torch.bool)
            gates = gate_circuit(projected_bits).reshape(1, 3, checkpoint["readout_bits"])
            counts = gates.sum(-1)
            expected = torch.cat([
                (counts[:, i:i + 1] + readout.beta) / readout.tau
                for i, readout in enumerate(model.readouts)
            ], dim=1)
            if not torch.allclose(core(projected_bits), expected, atol=1e-6):
                raise RuntimeError("Compact gate counts disagree with calibrated GroupSum")
        # The full-image and compact paths must agree on a valid ADC sample.
        codes = torch.randint(0, 4, (1, checkpoint["frames"], 16, 16))
        steps = torch.arange(1, 4).view(1, 1, 3, 1, 1)
        adc_bits = (codes.unsqueeze(2) >= steps).flatten(1, 2)
        proj_x, proj_y = charge_sum_projections(adc_bits, checkpoint["frames"])
        if not torch.allclose(model.forward_logic(adc_bits), core(torch.cat((proj_x, proj_y), dim=1)), atol=1e-6):
            raise RuntimeError("Compact input disagrees with full-image charge projection")
    circuit.simplify()
    # Constant gate outputs can be folded into beta during simplification.
    reductions = {sr.node_id: sr for sr in circuit.sum_nodes}
    output_reductions = [reductions.get(output) for output in circuit.outputs]
    if len(output_reductions) != len(model.readouts) or any(sr is None for sr in output_reductions):
        raise RuntimeError("Simplification changed the number of output reductions")

    # Verify that binary-input inference agrees after circuit lowering.
    torch.manual_seed(42)
    with torch.no_grad():
        for _ in range(3):
            bits = torch.randint(0, 2, (1, *input_shape), dtype=torch.bool)
            if not torch.allclose(core(bits), circuit(bits), atol=1e-6):
                raise RuntimeError("Exported circuit disagrees with hard-LUT PyTorch model")

    args.out.mkdir(parents=True, exist_ok=True)
    circuit.write_json(str(args.out / "circuit.json"))
    circuit.write_c_code(str(args.out / "circuit.c"))
    if compact:
        gate_circuit.simplify()
        gate_circuit.write_verilog_code(str(args.out / "gates.v"))
        (args.out / "compact_top.sv").write_text(
            count_rtl(checkpoint["readout_bits"], input_shape[0]))
    parameter_names = (["x", "y", "cotB"] if checkpoint["task"] == "slim" else
                       ["x", "y", "cotA", "cotB", "scale_x", "scale_y", "scale_cotA",
                        "scale_cotB"] + (["tril_y_x", "tril_cotA_x", "tril_cotA_y",
                                          "tril_cotB_x", "tril_cotB_y", "tril_cotB_cotA"]
                                         if checkpoint["task"] == "max" else []))
    metadata = {
        "checkpoint": str(args.checkpoint),
        "input_encoding": ("x-branch then y-branch; each (frame, coordinate) charge sum "
                           "has six least-significant-bit-first bits" if compact else
                           "2-bit ADC code -> three cumulative comparator bits per pixel and frame"),
        "binary_input_shape": list(input_shape),
        "external_input_bits": math.prod(input_shape),
        "projection_location": "upstream of compact circuit" if compact else "inside binary circuit",
        "count_output_bits": (3 * math.ceil(math.log2(checkpoint["readout_bits"] + 1))
                              if compact else None),
        "feature_names": parameter_names,
        "projection": checkpoint.get("projection", "or"),
        "circuit_gate_count": len(circuit.gates),
        "circuit_sum_nodes": len(circuit.sum_nodes),
        "group_sum_tau_beta": original_readouts,
        "compiled_tau_beta": [{"tau": sr.tau, "beta": sr.beta} for sr in output_reductions],
        "label_scale": checkpoint["label_scale"],
        "label_center": checkpoint.get("label_center", [0.0] * 4),
        "normalized_group_sum_bounds": (calibration["readout_bounds"] if calibration else None),
        "thresholds_electrons": checkpoint["data_metadata"]["thresholds_electrons"],
        "note": ("C/JSON include tau and beta. Simplification may fold constant gate outputs "
                 "into compiled beta while preserving the exact result. Current TorchLogix "
                 "Verilog does not apply scaling to nonconstant reductions. For the compact "
                 "Conv1D export, gates.v plus compact_top.sv emit exact unsigned counts; "
                 "apply each feature's tau, beta, center and scale downstream. "
                 "The upstream charge-sum projectors are not included in those RTL files."),
    }
    (args.out / "readout.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
