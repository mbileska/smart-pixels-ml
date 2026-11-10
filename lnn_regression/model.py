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


def add_binary_words(left, right):
    """A fixed ripple-carry adder built only from Boolean gate operations."""
    if len(left) != len(right):
        raise ValueError("Adder inputs must have equal widths")
    result = [left[0] ^ right[0]]
    carry = left[0] & right[0]
    for a, b in zip(left[1:], right[1:]):
        propagate = a ^ b
        result.append(propagate ^ carry)
        carry = (a & b) | (propagate & carry)
    return tuple(result + [carry])


def charge_sum_projections(bits, frames):
    """Exact per-axis sums of 2-bit ADC codes, encoded as six Boolean bits.

    Summing the 16 pixel codes and dividing by 16 is the paper's average
    projection. The division is a fixed scale and need not be implemented in
    the binary input to the learned logic gates. This Boolean adder tree keeps
    charge multiplicity that OR pooling discards and lowers to circuit gates.
    """
    batch = bits.shape[0]
    b = bits.bool().reshape(batch, frames, 3, 16, 16)
    # Thermometer code (>=1, >=2, >=3) to ordinary two-bit ADC code.
    low = b[:, :, 0] ^ b[:, :, 1] ^ b[:, :, 2]
    high = b[:, :, 1]

    def project(axis):
        words = []
        for i in range(16):
            words.append((low[:, :, :, i], high[:, :, :, i]) if axis == "x" else
                         (low[:, :, i, :], high[:, :, i, :]))
        while len(words) > 1:
            words = [add_binary_words(words[i], words[i + 1])
                     for i in range(0, len(words), 2)]
        projected = torch.cat([word.unsqueeze(-1) for word in words[0]], dim=-1).flatten(1)
        return projected if bits.dtype == torch.bool else projected.to(bits.dtype)

    return project("x"), project("y")


CHARGE_THRESHOLDS = (2, 4, 8, 16, 32)


def or_projections(bits):
    """Whether each ADC comparator fired anywhere on an axis."""
    if bits.dtype == torch.bool:
        proj_x = bits[:, :, :, 0]
        proj_y = bits[:, :, 0, :]
        for i in range(1, 16):
            proj_x = proj_x | bits[:, :, :, i]
            proj_y = proj_y | bits[:, :, i, :]
    else:
        proj_x = bits.amax(dim=3)
        proj_y = bits.amax(dim=2)
    return proj_x.flatten(1), proj_y.flatten(1)


def or_charge_threshold_projections(bits, frames):
    """Keep ADC OR signals and ordered threshold signals for axis charge sums.

    Each threshold is a power of two, so a six-bit sum reaches it exactly when
    any bit at or above its position is set. This lowers to Boolean OR gates.
    The sums are per time frame and per coordinate, as in the paper's pooling.
    """
    or_x, or_y = or_projections(bits)
    sum_x, sum_y = charge_sum_projections(bits, frames)
    batch = bits.shape[0]

    def combine(ors, sums):
        binary = sums.bool().reshape(batch, frames, 16, 6)
        thresholds = []
        for value in CHARGE_THRESHOLDS:
            bit = value.bit_length() - 1
            reached = binary[:, :, :, bit]
            for higher in range(bit + 1, 6):
                reached = reached | binary[:, :, :, higher]
            thresholds.append(reached.unsqueeze(2))
        ordered = torch.cat(thresholds, dim=2)
        if bits.dtype != torch.bool:
            ordered = ordered.to(bits.dtype)
        return torch.cat((ors.reshape(batch, frames, 3, 16), ordered), dim=2).flatten(1)

    return combine(or_x, sum_x), combine(or_y, sum_y)


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


class CompactChargeCircuitReadout(nn.Module):
    """Paper-style axis sums as a 384-bit input for two-frame circuit export.

    Each 16-pixel sum is represented by six bits, least significant bit first.
    X and Y branches are concatenated, with frames and coordinates in order.
    The upstream charge-sum projectors are outside this compact circuit.
    """

    def __init__(self, regressor):
        super().__init__()
        if regressor.projection != "charge_sum" or regressor.architecture != "conv1d":
            raise ValueError("Compact charge interface requires Conv1D charge_sum")
        self.regressor = regressor

    def forward(self, projected_bits):
        branch_width = self.regressor.frames * 16 * 6
        assert projected_bits.ndim == 2 and projected_bits.shape[1] == 2 * branch_width
        return self.regressor.forward_projected_logic(
            projected_bits[:, :branch_width], projected_bits[:, branch_width:])


class LogicRegressor(nn.Module):
    """Paper-shaped MLP, Conv1D and Conv2D alternatives using Boolean gates.

    The ADC levels are encoded as three cumulative comparator bits. All learned
    operations are LUT gates. Fixed Boolean projection or pooling feeds the
    learned network; the output decoder counts binary gates. There are no
    trainable floating-point dense or convolution layers.
    """

    def __init__(self, architecture="mlp", task="slim", frames=2, readout_bits=64,
                 device="cpu", readout_bounds=None, projection="or"):
        super().__init__()
        if architecture not in ("mlp", "conv1d", "conv2d"):
            raise ValueError(architecture)
        if task not in ("slim", "full", "max"):
            raise ValueError(task)
        if task == "max" and architecture != "conv2d":
            raise ValueError("The paper uses Max only with Conv2D")
        if projection not in ("or", "charge_sum", "or_charge_thresholds") or (
                projection == "charge_sum" and architecture == "conv2d") or (
                projection == "or_charge_thresholds" and architecture != "mlp"):
            raise ValueError("This charge projection is unsupported for the architecture")
        self.architecture, self.task, self.frames = architecture, task, frames
        self.projection = projection
        if readout_bits < 1:
            raise ValueError("readout_bits must be positive")
        self.readout_bits = readout_bits
        n_raw = {"slim": 3, "full": 8, "max": 14}[task]
        gate = lambda a, b: LogicDense(a, b, device=device,
                                      parametrization_kwargs={"forward_sampling": "hard"})
        channels = 3 * frames
        if architecture == "mlp":
            projection_width = 16 * ({"or": 3, "charge_sum": 6,
                                      "or_charge_thresholds": 3 + len(CHARGE_THRESHOLDS)}[projection] * frames)
            self.proj_x = reduce_gates(projection_width, 16, device)
            self.proj_y = reduce_gates(projection_width, 16, device)
            self.embedding = gate(32, 16)
            self.hidden1 = gate(16, 16)
            self.hidden2 = gate(16, 16)
            head_in = 16
        elif architecture == "conv1d":
            conv_channels = 6 * frames if projection == "charge_sum" else channels
            tree_depth = 6 if projection == "charge_sum" else 2
            self.proj_x = LogicConv2d(in_dim=(16, 1), channels=conv_channels, num_kernels=5,
                                      tree_depth=tree_depth, receptive_field_size=(3, 1), device=device,
                                      parametrization_kwargs={"forward_sampling": "hard"})
            self.proj_y = LogicConv2d(in_dim=(16, 1), channels=conv_channels, num_kernels=5,
                                      tree_depth=tree_depth, receptive_field_size=(3, 1), device=device,
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
        # Calibration supplies independently fitted intervals for each feature.
        # Legacy defaults permit older checkpoints to remain evaluable.
        if readout_bounds is None:
            readout_bounds = {"mean": [[-2.0, 2.0]] * 4,
                              "sigma": [[0.02, 2.02]] * 4,
                              "offdiag": [[-1.0, 1.0]] * 6}
        mean_indices = (0, 1, 3) if task == "slim" else (0, 1, 2, 3)
        bounds = [readout_bounds["mean"][i] for i in mean_indices]
        if task != "slim":
            bounds += list(readout_bounds["sigma"])
        if task == "max":
            bounds += list(readout_bounds["offdiag"])
        if len(bounds) != n_raw:
            raise ValueError("Readout bounds do not match output count")
        readout_specs = []
        for low, high in bounds:
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"Invalid GroupSum bounds: {low}, {high}")
            tau = readout_bits / (high - low)
            readout_specs.append((tau, low * tau))
        self.readouts = nn.ModuleList(
            GroupSum(k=1, tau=tau, beta=beta, device=device)
            for tau, beta in readout_specs
        )

    def forward_logic(self, bits):
        """Evaluate gate network on thermometer-encoded binary ADC inputs."""
        assert bits.ndim == 4 and bits.shape[1:] == (3 * self.frames, 16, 16)
        if self.architecture in ("mlp", "conv1d"):
            if self.projection == "charge_sum":
                proj_x, proj_y = charge_sum_projections(bits, self.frames)
            elif self.architecture == "mlp" and self.projection == "or_charge_thresholds":
                proj_x, proj_y = or_charge_threshold_projections(bits, self.frames)
            else:
                # Legacy OR projection retained for existing checkpoints.
                proj_x, proj_y = or_projections(bits)
            return self.forward_projected_logic(proj_x, proj_y)
        else:
            z = self.pool(self.conv_pointwise(self.conv_spatial(bits))).flatten(1)
        return self._readout(z)

    def forward_projected_logic(self, proj_x, proj_y):
        """Evaluate MLP or Conv1D after its fixed axis projection."""
        if self.architecture == "mlp":
            x = self.proj_x(proj_x)
            y = self.proj_y(proj_y)
            z = self.embedding(torch.cat((x, y), dim=1))
        elif self.architecture == "conv1d":
            channels = (6 if self.projection == "charge_sum" else 3) * self.frames
            if self.projection == "charge_sum":
                # Sum bits arrive as (frame, coordinate, bit); convolution
                # expects (frame, bit, coordinate). Keep the 16-pixel axis
                # as its spatial dimension, matching the paper's Conv1D.
                x_input = proj_x.reshape(-1, self.frames, 16, 6).permute(0, 1, 3, 2)
                y_input = proj_y.reshape(-1, self.frames, 16, 6).permute(0, 1, 3, 2)
            else:
                x_input = proj_x.reshape(-1, channels, 16)
                y_input = proj_y.reshape(-1, channels, 16)
            x = self.proj_x(x_input.reshape(-1, channels, 16, 1))
            y = self.proj_y(y_input.reshape(-1, channels, 16, 1))
            z = torch.cat((x.flatten(1), y.flatten(1)), dim=1)
        else:
            raise ValueError("Only projected architectures can use this entry point")
        return self._readout(z)

    def _readout(self, z):
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
