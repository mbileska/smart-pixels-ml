"""Pure TorchLogix gate networks with a fixed count-to-physics readout."""

import math

import torch
from torch import nn
from torchlogix.layers import GroupSum, LogicConv2d, LogicDense, OrPooling2d


def reduce_gates(input_width, output_width, device):
    """Binary fan-in requires staged reductions to cover every input wire."""
    layers = []
    width = input_width
    while width > 2 * output_width:
        next_width = (width + 1) // 2
        layers.append(LogicDense(width, next_width, device=device,
                                 parametrization_kwargs={"forward_sampling": "hard"}))
        width = next_width
    layers.append(LogicDense(width, output_width, device=device,
                             parametrization_kwargs={"forward_sampling": "hard"}))
    return nn.Sequential(*layers)


class CircuitReadout(nn.Module):
    """Binary-input gate core for TorchLogix Circuit export.

    The fixed ADC comparator encoding is upstream of this module. Its outputs
    are the normalized regression parameters, with tau/beta in GroupSum nodes.
    """

    def __init__(self, regressor):
        super().__init__()
        self.regressor = regressor

    def forward(self, bits):
        return self.regressor.forward_logic(bits)


class LogicRegressor(nn.Module):
    """Paper-shaped MLP, Conv1D and Conv2D alternatives using Boolean gates.

    The ADC levels are encoded as three cumulative comparator bits. All learned
    operations are LUT gates. Pooling is OR, and the fixed output decoder counts
    binary gates; there are no trainable floating-point dense or convolution layers.
    """

    def __init__(self, architecture="mlp", task="slim", frames=2, readout_bits=64, device="cpu"):
        super().__init__()
        if architecture not in ("mlp", "conv1d", "conv2d"):
            raise ValueError(architecture)
        if task not in ("slim", "full", "max"):
            raise ValueError(task)
        if task == "max" and architecture != "conv2d":
            raise ValueError("The paper uses Max only with Conv2D")
        self.architecture, self.task, self.frames = architecture, task, frames
        if readout_bits < 1:
            raise ValueError("readout_bits must be positive")
        self.readout_bits = readout_bits
        n_raw = {"slim": 3, "full": 8, "max": 14}[task]
        gate = lambda a, b: LogicDense(a, b, device=device,
                                      parametrization_kwargs={"forward_sampling": "hard"})
        channels = 3 * frames
        if architecture == "mlp":
            self.proj_x = reduce_gates(16 * channels, 16, device)
            self.proj_y = reduce_gates(16 * channels, 16, device)
            self.embedding = gate(32, 16)
            self.hidden1 = gate(16, 16)
            self.hidden2 = gate(16, 16)
            head_in = 16
        elif architecture == "conv1d":
            self.proj_x = LogicConv2d(in_dim=(16, 1), channels=channels, num_kernels=5,
                                      tree_depth=2, receptive_field_size=(3, 1), device=device,
                                      parametrization_kwargs={"forward_sampling": "hard"})
            self.proj_y = LogicConv2d(in_dim=(16, 1), channels=channels, num_kernels=5,
                                      tree_depth=2, receptive_field_size=(3, 1), device=device,
                                      parametrization_kwargs={"forward_sampling": "hard"})
            self.hidden1 = reduce_gates(140, 16, device)
            self.hidden2 = gate(16, 16)
            head_in = 16
        else:
            self.conv_spatial = LogicConv2d(in_dim=(16, 16), channels=channels, num_kernels=5,
                                            tree_depth=3, receptive_field_size=3, device=device,
                                            parametrization_kwargs={"forward_sampling": "hard"})
            self.conv_pointwise = LogicConv2d(in_dim=(14, 14), channels=5, num_kernels=5,
                                              tree_depth=2, receptive_field_size=1, device=device,
                                              parametrization_kwargs={"forward_sampling": "hard"})
            self.pool = OrPooling2d(kernel_size=3, stride=3)
            self.hidden1 = reduce_gates(80, 16, device)
            self.hidden2 = gate(16, 16)
            head_in = 16
        # Random output LUTs start the count near one half; residual LUTs would
        # start all means at -2 and Max off-diagonals at -1.
        self.output_gates = LogicDense(head_in, n_raw * readout_bits, device=device,
                                       parametrization_kwargs={"forward_sampling": "hard",
                                                               "weight_init": "random"})
        # One independent sum per physical output. GroupSum owns the scale and
        # offset so TorchLogix preserves them in Circuit sum_nodes and C code.
        mean_count = 3 if task == "slim" else 4
        readout_specs = ([(readout_bits / 4, -readout_bits / 2)] * mean_count)
        if task != "slim":
            readout_specs += [(readout_bits / 2, 0.01 * readout_bits)] * 4
        if task == "max":
            readout_specs += [(readout_bits / 2, -readout_bits / 2)] * 6
        self.readouts = nn.ModuleList(
            GroupSum(k=1, tau=tau, beta=beta, device=device)
            for tau, beta in readout_specs
        )

    def forward_logic(self, bits):
        """Evaluate gate network on thermometer-encoded binary ADC inputs."""
        assert bits.ndim == 4 and bits.shape[1:] == (3 * self.frames, 16, 16)
        if self.architecture in ("mlp", "conv1d"):
            # Boolean projection counterpart of the paper's charge average pooling.
            if bits.dtype == torch.bool:
                proj_x = bits[:, :, :, 0]
                proj_y = bits[:, :, 0, :]
                for i in range(1, 16):
                    proj_x = proj_x | bits[:, :, :, i]
                    proj_y = proj_y | bits[:, :, i, :]
            else:
                proj_x = bits.amax(dim=3)
                proj_y = bits.amax(dim=2)
            proj_x, proj_y = proj_x.flatten(1), proj_y.flatten(1)
            if self.architecture == "mlp":
                x = self.proj_x(proj_x)
                y = self.proj_y(proj_y)
                z = self.embedding(torch.cat((x, y), dim=1))
            else:
                x = self.proj_x(proj_x.reshape(-1, 3 * self.frames, 16, 1))
                y = self.proj_y(proj_y.reshape(-1, 3 * self.frames, 16, 1))
                z = torch.cat((x.flatten(1), y.flatten(1)), dim=1)
        else:
            z = self.pool(self.conv_pointwise(self.conv_spatial(bits))).flatten(1)
        z = self.hidden2(self.hidden1(z))
        gates = self.output_gates(z)
        return torch.cat([
            readout(gates[:, i * self.readout_bits:(i + 1) * self.readout_bits])
            for i, readout in enumerate(self.readouts)
        ], dim=1)

    def forward(self, codes):
        assert codes.ndim == 4 and codes.shape[1:] == (self.frames, 16, 16)
        steps = torch.arange(1, 4, device=codes.device).view(1, 1, 3, 1, 1)
        bits = (codes.unsqueeze(2) >= steps).float().flatten(1, 2)
        raw = self.forward_logic(bits)
        n_mean = 3 if self.task == "slim" else 4
        means = raw[:, :n_mean]
        if self.task == "slim":
            return {"mean": means}
        # Positive widths and Cholesky diagonal; units are normalized truth units.
        sigma = raw[:, n_mean:2 * n_mean]
        if self.task == "full":
            return {"mean": means, "sigma": sigma}
        lower = raw[:, 8:]
        tril = torch.diag_embed(sigma)
        tril[:, 1, 0] = lower[:, 0]
        tril[:, 2, 0] = lower[:, 1]
        tril[:, 2, 1] = lower[:, 2]
        tril[:, 3, 0] = lower[:, 3]
        tril[:, 3, 1] = lower[:, 4]
        tril[:, 3, 2] = lower[:, 5]
        return {"mean": means, "sigma": torch.linalg.vector_norm(tril, dim=-1), "scale_tril": tril}


def regression_loss(pred, target, task):
    if task == "slim":
        return torch.mean((pred["mean"] - target[:, (0, 1, 3)]) ** 2)
    if task == "full":
        scaled = (target - pred["mean"]) / pred["sigma"]
        return (0.5 * scaled.square() + torch.log(pred["sigma"]) +
                0.5 * math.log(2 * math.pi)).sum(dim=1).mean()
    delta = (target - pred["mean"]).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(pred["scale_tril"], delta, upper=False).squeeze(-1)
    logdet = torch.log(torch.diagonal(pred["scale_tril"], dim1=-2, dim2=-1)).sum(dim=1)
    return (0.5 * whitened.square().sum(dim=1) + logdet + 2 * math.log(2 * math.pi)).mean()
