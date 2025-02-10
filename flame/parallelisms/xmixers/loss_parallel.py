# TODO: still has bug in bwd, update this later
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import (DeviceMesh, DTensor, Partial, Replicate,
                                      Shard, distribute_module,
                                      distribute_tensor)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement


class LossParallel(ParallelStyle):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = False,
    ):
        super().__init__()
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output
        self.input_layouts = input_layouts
        self.desired_input_layouts = (Shard(1),)
        self.output_layouts = output_layouts

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh
    ):
        (
            weight,
            hidden_states,
            labels,
            bias,
            num_logits_to_keep,
            fuse_linear_and_cross_entropy,
        ) = inputs
        if not isinstance(weight, DTensor):
            weight = DTensor.from_local(weight, device_mesh, [Replicate()])

        if bias is not None and not isinstance(bias, DTensor):
            bias = DTensor.from_local(bias, device_mesh, [Replicate()])

        if not isinstance(hidden_states, DTensor):
            hidden_states = DTensor.from_local(
                hidden_states, device_mesh, input_layouts
            )

        if hidden_states.placements != desired_input_layouts:
            hidden_states = hidden_states.redistribute(
                placements=desired_input_layouts, async_op=True
            )

        if not isinstance(labels, DTensor):
            labels = DTensor.from_local(labels, device_mesh, [Replicate()])

        if labels.placements != desired_input_layouts:
            labels = labels.redistribute(
                placements=desired_input_layouts, async_op=True
            )

        return (
            weight.to_local(),
            hidden_states.to_local(),
            labels.to_local(),
            bias.to_local() if bias is not None else bias,
            num_logits_to_keep,
            fuse_linear_and_cross_entropy,
        )

    def _partition_fn(self, name, module, device_mesh):
        pass

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        if not isinstance(output1, DTensor) and output1 is not None:
            output1 = DTensor.from_local(output1, device_mesh, [Replicate()])

        return output1, output2

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
            partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            partial(
                self._prepare_output_fn, self.output_layouts, self.use_local_output
            ),
        )
