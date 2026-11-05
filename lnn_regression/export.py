"""Export the binary LNN core with one calibrated sum reduction per feature."""

import argparse
import json
from pathlib import Path

import torch

from .model import CircuitReadout, LogicRegressor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    from torchlogix import Circuit

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LogicRegressor(checkpoint["architecture"], checkpoint["task"],
                           checkpoint["frames"], checkpoint["readout_bits"], "cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    core = CircuitReadout(model)
    input_shape = (3 * checkpoint["frames"], 16, 16)
    circuit = Circuit.from_model(core, input_shape=input_shape)
    reductions = {sr.node_id: sr for sr in circuit.sum_nodes}
    output_reductions = [reductions.get(output) for output in circuit.outputs]
    if len(output_reductions) != len(model.readouts) or any(sr is None for sr in output_reductions):
        raise RuntimeError("Expected exactly one Circuit sum reduction per output feature")
    for i, (sr, group) in enumerate(zip(output_reductions, model.readouts)):
        if sr.tau != group.tau or abs(sr.beta - group.beta) > 1e-6:
            raise RuntimeError(f"Feature {i} lost GroupSum tau/beta during export")
    original_readouts = [{"tau": group.tau, "beta": group.beta} for group in model.readouts]
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
    parameter_names = (["x", "y", "cotB"] if checkpoint["task"] == "slim" else
                       ["x", "y", "cotA", "cotB", "scale_x", "scale_y", "scale_cotA",
                        "scale_cotB"] + (["tril_y_x", "tril_cotA_x", "tril_cotA_y",
                                          "tril_cotB_x", "tril_cotB_y", "tril_cotB_cotA"]
                                         if checkpoint["task"] == "max" else []))
    metadata = {
        "checkpoint": str(args.checkpoint),
        "input_encoding": "2-bit ADC code -> three cumulative comparator bits per pixel and frame",
        "binary_input_shape": list(input_shape),
        "feature_names": parameter_names,
        "group_sum_tau_beta": original_readouts,
        "compiled_tau_beta": [{"tau": sr.tau, "beta": sr.beta} for sr in output_reductions],
        "label_scale": checkpoint["label_scale"],
        "thresholds_electrons": checkpoint["data_metadata"]["thresholds_electrons"],
        "note": ("C/JSON include tau and beta. Simplification may fold constant gate outputs "
                 "into compiled beta while preserving the exact result. Current TorchLogix "
                 "Verilog does not apply scaling to nonconstant reductions."),
    }
    (args.out / "readout.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
