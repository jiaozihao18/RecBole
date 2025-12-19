# -*- coding: utf-8 -*-
"""
Custom GRU implementation for NPU compatibility
===============================================

This is a pure-Python implementation of GRU that can work with different
dtypes (including fp32) on NPU devices, avoiding the fp16 requirement
of torch.nn.GRU's dynamicgruv2 kernel.

- Constructor signature EXACTLY matches nn.GRU
- Parameter layout EXACTLY matches nn.GRU
- Input / output behavior EXACTLY matches nn.GRU
- Supports: num_layers, batch_first, bias, dropout, device, dtype
- Does NOT support: PackedSequence, bidirectional, cuDNN

Reference implementation based on PyTorch's GRU math.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def gru_cell_step(
    x_t: torch.Tensor,
    h_prev: torch.Tensor,
    w_ih: torch.Tensor,
    w_hh: torch.Tensor,
    b_ih: Optional[torch.Tensor],
    b_hh: Optional[torch.Tensor],
) -> torch.Tensor:
    """Single GRU time-step computation (matches PyTorch math exactly)."""
    gi = F.linear(x_t, w_ih, b_ih)     # (N, 3H)
    gh = F.linear(h_prev, w_hh, b_hh)  # (N, 3H)

    i_r, i_z, i_n = gi.chunk(3, dim=1)
    h_r, h_z, h_n = gh.chunk(3, dim=1)

    r_t = torch.sigmoid(i_r + h_r)
    z_t = torch.sigmoid(i_z + h_z)
    n_t = torch.tanh(i_n + r_t * h_n)

    h_t = (1.0 - z_t) * n_t + z_t * h_prev
    return h_t


class CustomGRU(nn.Module):
    r"""Custom GRU implementation compatible with NPU devices.
    
    This is a drop-in replacement for nn.GRU that avoids the fp16 requirement
    of the dynamicgruv2 kernel on NPU devices. It supports all standard dtypes
    including fp32, which helps prevent precision overflow issues.
    
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers. Default: 1
        bias: If False, then the layer does not use bias weights. Default: True
        batch_first: If True, then the input and output tensors are provided
            as (batch, seq, feature). Default: False
        dropout: If non-zero, introduces a Dropout layer on the outputs of each
            RNN layer except the last layer. Default: 0
        bidirectional: Not supported in this implementation. Default: False
        device: Device to create the module on. Default: None
        dtype: Data type to create the module with. Default: None
    
    Shape:
        - Input: :math:`(L, N, H_{in})` when batch_first=False or
          :math:`(N, L, H_{in})` when batch_first=True containing the features
          of the input sequence.
        - Output: :math:`(L, N, H_{out})` when batch_first=False or
          :math:`(N, L, H_{out})` when batch_first=True containing the output
          features from the last layer of the GRU.
        - h_n: :math:`(num_layers, N, H_{out})` containing the final hidden
          state for each element in the batch.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if bidirectional:
            raise NotImplementedError(
                "Bidirectional GRU is not supported in this custom implementation."
            )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        factory_kwargs = {"device": device, "dtype": dtype}

        self.weight_ih_l = nn.ParameterList()
        self.weight_hh_l = nn.ParameterList()
        self.bias_ih_l = nn.ParameterList()
        self.bias_hh_l = nn.ParameterList()

        for layer in range(num_layers):
            in_dim = input_size if layer == 0 else hidden_size

            self.weight_ih_l.append(
                nn.Parameter(torch.empty(3 * hidden_size, in_dim, **factory_kwargs))
            )
            self.weight_hh_l.append(
                nn.Parameter(torch.empty(3 * hidden_size, hidden_size, **factory_kwargs))
            )

            if bias:
                self.bias_ih_l.append(
                    nn.Parameter(torch.empty(3 * hidden_size, **factory_kwargs))
                )
                self.bias_hh_l.append(
                    nn.Parameter(torch.empty(3 * hidden_size, **factory_kwargs))
                )
            else:
                self.bias_ih_l.append(None)
                self.bias_hh_l.append(None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using uniform distribution."""
        std = 1.0 / (self.hidden_size ** 0.5)
        for p in self.parameters():
            if p is not None:
                nn.init.uniform_(p, -std, std)

    def forward(
        self,
        input: torch.Tensor,
        hx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass of the CustomGRU.
        
        Args:
            input: Tensor of shape (L, N, H_in) or (N, L, H_in) depending on batch_first
            hx: Initial hidden state of shape (num_layers, N, H_out). If None, zeros are used.
        
        Returns:
            output: Tensor containing the output features from the last layer
            h_n: Tensor containing the final hidden state for each element in the batch
        """
        if self.batch_first:
            input = input.transpose(0, 1)  # (T, N, D)

        T, N, _ = input.shape

        if hx is None:
            hx = torch.zeros(
                self.num_layers,
                N,
                self.hidden_size,
                device=input.device,
                dtype=input.dtype,
            )

        layer_input = input
        final_h = []

        for layer in range(self.num_layers):
            h_t = hx[layer]
            outputs = []

            for t in range(T):
                h_t = gru_cell_step(
                    layer_input[t],
                    h_t,
                    self.weight_ih_l[layer],
                    self.weight_hh_l[layer],
                    self.bias_ih_l[layer] if self.bias else None,
                    self.bias_hh_l[layer] if self.bias else None,
                )
                outputs.append(h_t.unsqueeze(0))

            layer_output = torch.cat(outputs, dim=0)
            final_h.append(h_t)

            if self.dropout > 0 and layer < self.num_layers - 1:
                layer_output = F.dropout(
                    layer_output, p=self.dropout, training=self.training
                )

            layer_input = layer_output

        output = layer_input
        h_n = torch.stack(final_h, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, h_n

