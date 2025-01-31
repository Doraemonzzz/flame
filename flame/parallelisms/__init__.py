from typing import Callable, Union

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (CPUOffloadPolicy,
                                                MixedPrecisionPolicy,
                                                fully_shard)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.pipelining import PipelineStage
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)
from torchtitan.config_manager import TORCH_DTYPE_MAP, JobConfig
from torchtitan.logging import logger
from torchtitan.parallelisms.parallel_dims import ParallelDims
from torchtitan.parallelisms.pipelining_utils import (build_pipeline_schedule,
                                                      generate_split_points,
                                                      stage_ids_this_rank)
from transformers import PretrainedConfig

from .fla import parallelize_fla, pipeline_fla
from .xmixers import parallelize_xmixers, pipeline_xmixers

DeviceType = Union[int, str, torch.device]


def pipeline_model(
    model: nn.Module,
    pp_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    device: DeviceType,
    model_config: PretrainedConfig,
    loss_fn: Callable[..., torch.Tensor],
    model_source: str = "xmixers",
):
    if model_source == "fla":
        return pipeline_fla(
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )
    elif model_source == "xmixers":
        return pipeline_xmixers(
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )


def parallelize_model(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
    model_source: str = "xmixers",
):
    if model_source == "fla":
        return parallelize_fla(model, world_mesh, parallel_dims, job_config)
    elif model_source == "xmixers":
        return parallelize_xmixers(model, world_mesh, parallel_dims, job_config)
