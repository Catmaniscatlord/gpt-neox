import torch
import torch.nn as nn

from .utils import get_parallel_linear


class GRUInit(nn.Module):
    """Standalone GRU module — exists only to be weight-tied."""

    def __init__(self, neox_args):
        super().__init__()
        self.neox_args = neox_args

    def forward(self, args):
        hidden_states, attention_mask = args
        s, b, h = hidden_states.shape
        gru_states = torch.zeros(
            (s, b, self.neox_args.gru_width),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return gru_states, hidden_states, attention_mask


class GRULayer(nn.Module):
    """Standalone GRU module — exists only to be weight-tied."""

    def __init__(self, neox_args):
        super().__init__()
        self.neox_args = neox_args

        ColumnParallelLinear, RowParallelLinear = get_parallel_linear(neox_args)

        self.W = ColumnParallelLinear(
            neox_args=neox_args,
            input_size=(neox_args.hidden_size + neox_args.gru_width),
            output_size=3 * neox_args.gru_width,
            bias=neox_args.gru_use_bias,
        )

    def forward(self, hidden_states, gru_states):
        xh = torch.cat((hidden_states, gru_states), dim=-1)
        z, r, h_hat = self.W(xh)[0].chunk(3, dim=-1)

        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        h_hat = torch.tanh(h_hat)
        gru_states = (1 - z) * gru_states + z * h_hat

        return gru_states


class GRUOut(nn.Module):
    """Standalone GRU module — exists only to be weight-tied."""

    def __init__(self, neox_args):
        super().__init__()
        self.neox_args = neox_args

        ColumnParallelLinear, RowParallelLinear = get_parallel_linear(neox_args)

        self.W = RowParallelLinear(
            neox_args=neox_args,
            input_size=neox_args.hidden_size + neox_args.gru_width,
            output_size=neox_args.hidden_size,
            bias=neox_args.gru_use_bias,
        )

    def forward(self, hidden_states, gru_states):
        xh = torch.cat((hidden_states, gru_states), dim=-1)
        return self.W(xh)[0]


class GRULayerPipe(GRULayer):
    def forward(self, args):
        assert (
            len(args) == 3
        ), "GRULayerPipe expects 3 arguments - gru_states, hidden_states, and attention_mask"
        gru_states, hidden_states, attention_mask = args
        # we are returning just [gru_states, hidden_states, mask]
        return super().forward(hidden_states, gru_states), hidden_states, attention_mask


class GRUOutPipe(GRUOut):
    def forward(self, args):
        assert (
            len(args) == 3
        ), "GRUOutPipe expects 3 arguments - gru_states, hidden_states, and attention_mask"
        gru_states, hidden_states, attention_mask = args
        # we are returning just [gru_states, hidden_states, mask]
        return super().forward(hidden_states, gru_states), attention_mask


class GRULayerWrapperPipe(nn.Module):
    def __init__(
        self,
        neox_args,
        block_cls,
        **block_kwargs,
    ):
        super().__init__()

        self.is_gru = neox_args.use_gru

        self.layer = block_cls(neox_args=neox_args, **block_kwargs)

    def forward(self, args):
        if self.is_gru:
            assert (
                len(args) == 3
            ), "LayerWrapperPipe expects 3 arguments - gru_states, hidden_states, and attention_mask"
            gru_states, hidden_states, attention_mask = args
            # we are returning just [hidden_states, mask]
            return (
                gru_states,
                self.layer.forward((hidden_states, attention_mask))[0],
                attention_mask,
            )
        else:
            assert (
                len(args) == 2
            ), "LayerWrapperPipe expects 2 arguments - hidden_states and attention_mask"
            return self.layer.forward(args)
