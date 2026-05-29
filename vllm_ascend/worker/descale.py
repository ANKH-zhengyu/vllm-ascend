import gc
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch_npu
from torch.distributed.distributed_c10d import _set_pg_timeout
from vllm.compilation.wrapper import reset_compile_wrapper
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tp_group,
    stateless_init_torch_distributed_process_group,
)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig
from vllm.model_executor.model_loader import get_model_loader

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import (
    destroy_ascend_model_parallel,
    get_dynamic_eplb_group,
    get_mc2_group,
    set_elastic_info,
)
from vllm_ascend.eplb.core.eplb_utils import generate_log2phy_map, init_eplb_config
from vllm_ascend.ops.fused_moe.fused_moe import setup_moe_comm_method

if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.worker.worker import NPUWorker
else:
    NPUModelRunner = None
    NPUWorker = None


# TODO: Refactor descale.py - use descaler object instead of NpuWorker attrs to streamline code


def gen_expert_backup_map(
    num_experts: int, ep_size: int, num_die_per_npu: int, global_expert_distribution: dict[int, list[int]]
) -> list[list[int]]:
    backup_experts = [[] for _ in range(ep_size)]
    if global_expert_distribution is None:
        global_expert_distribution = distribute_experts(num_experts, ep_size)

    def get_least_load_backup_rank(exclude_ranks: list[int]) -> int:
        assert len(exclude_ranks) != ep_size, "At least one backup rank must remain available."
        min_backup_count = float("inf")
        optimal_backup_rank = -1
        for rank in range(ep_size):
            if rank in exclude_ranks:
                continue
            current_backup_count = len(backup_experts[rank])
            if current_backup_count <= min_backup_count:
                min_backup_count = current_backup_count
                optimal_backup_rank = rank
        return optimal_backup_rank

    for rank_group_start in range(0, ep_size, num_die_per_npu):
        rank_group_end = min(ep_size, rank_group_start + num_die_per_npu)
        current_rank_group = list(range(rank_group_start, rank_group_end))

        current_group_experts = []
        for rank in current_rank_group:
            current_group_experts.extend(global_expert_distribution[rank])
        for expert_id in current_group_experts:
            backup_rank = get_least_load_backup_rank(current_rank_group)
            backup_experts[backup_rank].append(expert_id)
    return backup_experts


def distribute_experts(global_num_expert: int, ep_size: int) -> dict[int, list[int]]:
    init_global_expert_distribution = {}
    base = global_num_expert // ep_size
    remainder = global_num_expert % ep_size

    start_index = 0
    for rank in range(ep_size):
        num = base + (1 if rank < remainder else 0)
        expert_ids = list(range(start_index, start_index + num))
        init_global_expert_distribution[rank] = expert_ids
        start_index += num
    return init_global_expert_distribution


def gen_global_log2phy_map(
    num_logical_experts: int, num_npu: int, redundant_expert_list: list[int]
) -> dict[int, list[int]]:
    num_redundant_experts = len(redundant_expert_list)
    assert (num_logical_experts + num_redundant_experts) % num_npu == 0, (
        "the physical expert count must evenly divide across NPUs"
    )
    num_phy_exp_per_npu = (num_logical_experts + num_redundant_experts) // num_npu

    # How many physical experts per NPU after placing redundancy
    exp_distribution_without_redundancy = distribute_experts(num_logical_experts, num_npu)
    num_routed_experts_list = []
    num_redundant_experts_list = []
    for rank in range(num_npu):
        num_routed_experts_list.append(len(exp_distribution_without_redundancy[rank]))
        num_redundant_experts_list.append(num_phy_exp_per_npu - len(exp_distribution_without_redundancy[rank]))

    # Mapping: logical expert -> list of physical expert IDs assigned
    global_log2phy_map: dict[int, list[int]] = {log_expert_id: [] for log_expert_id in range(num_logical_experts)}
    log_experts_iter = iter(range(num_logical_experts))

    global_pos = 0
    re_exp_assign_map = [[exp_id, False] for exp_id in redundant_expert_list]

    for rank in range(num_npu):
        local_expert_map = []
        for _ in range(num_routed_experts_list[rank]):
            expert_id = next(log_experts_iter)
            global_log2phy_map[expert_id].insert(0, global_pos)
            global_pos += 1
            local_expert_map.append(expert_id)

        for _ in range(num_redundant_experts_list[rank]):
            success = False
            for i in range(len(re_exp_assign_map)):
                eid, assigned = re_exp_assign_map[i]
                if assigned:
                    continue
                if eid in local_expert_map:
                    continue
                global_log2phy_map[eid].append(global_pos)
                global_pos += 1
                local_expert_map.append(eid)
                re_exp_assign_map[i][1] = True
                success = True
                break
            if not success:
                raise RuntimeError(
                    "expert placement aborted. The distribution of redundant experts cannot"
                    "satisfy the requirement that physical replicas of each logical expert are properly replicated."
                )
    return global_log2phy_map


def init_global_expert_distribution(global_log2phy_map: dict[int, list[int]], ep_size: int) -> dict[int, list[int]]:
    num_phy_experts = sum(map(len, global_log2phy_map.values()))
    num_phy_exp_per_npu = num_phy_experts // ep_size
    global_expert_distribution = {i: [-1 for _ in range(num_phy_exp_per_npu)] for i in range(ep_size)}
    for log_eid, phy_expert_pos in global_log2phy_map.items():
        for pos in phy_expert_pos:
            rank = pos // num_phy_exp_per_npu
            local_pos = pos - rank * num_phy_exp_per_npu
            global_expert_distribution[rank][local_pos] = log_eid
    return global_expert_distribution


def generate_redundant_expert_ids(num_experts: int, ep_size: int, num_redundant_experts: int) -> list[int]:
    assert num_redundant_experts % ep_size == 0
    experts_per_ep_group = num_experts // ep_size
    redundant_per_group = num_redundant_experts // ep_size
    redundant_ids = []
    for rank in range(ep_size):
        start_id = rank * experts_per_ep_group
        for i in range(redundant_per_group):
            redundant_ids.append(start_id + i)
    return redundant_ids


def get_expert_distribution_after_descale(
    model_runner,
    exclued_dp_ranks,
    enable_d2d_after_failure,
    rank_id,
):
    eplb_updator = model_runner.eplb_updator
    model_runner.shared_dict["descale"] = True
    model_runner.shared_dict["enable_d2d_after_failure"] = enable_d2d_after_failure
    model_runner.shared_dict["excluded_dp_ranks"] = exclued_dp_ranks
    if model_runner.shared_dict["expert_maps"] is None and model_runner.shared_dict["expert_maps"]:
        model_runner.shared_dict["expert_maps"] = get_global_expert_map(model_runner)

    eplb_updator.wakeup_eplb_worker()
    eplb_updator.update_info_all = eplb_updator.eplb_process.block_update_q.get()
    need_load_h2d = model_runner.shared_dict["need_load_h2d"]

    cur_rank_need_load_h2d = []
    for layer_id in range(len(need_load_h2d)):
        cur_rank_need_load = need_load_h2d[layer_id][rank_id].copy()
        cur_rank_need_load_h2d.append(cur_rank_need_load)

    # Append data for MTP layers (same distribution as corresponding main model layers)
    vllm_config = getattr(model_runner, "vllm_config", None)
    num_mtp_layers = _get_mtp_num_layers(vllm_config) if vllm_config is not None else 0
    if num_mtp_layers > 0:
        num_main_moe_layers = len(need_load_h2d)
        for mtp_layer_idx in range(num_mtp_layers):
            main_layer_idx = mtp_layer_idx % num_main_moe_layers if num_main_moe_layers > 0 else 0
            cur_rank_need_load_h2d.append(need_load_h2d[main_layer_idx][rank_id].copy())

    return cur_rank_need_load_h2d


def destroy_acl_graph(use_mask_mc2: bool, vllm_config: VllmConfig, model: NPUModelRunner) -> VllmConfig:
    if not use_mask_mc2:
        for entries in vllm_config.compilation_config.concrete_aclgraph_entry_list:
            for name, entry in entries.items():
                entry.aclgraph.reset()
                entry.aclgraph = None
        vllm_config.compilation_config.concrete_aclgraph_entry_list = []

        with set_current_vllm_config(vllm_config):
            reset_compile_wrapper(model.get_model())
        gc.collect()
        torch.npu.empty_cache()
    return vllm_config


def rebuild_acl_graph(use_mask_mc2: bool, worker: NPUWorker) -> None:
    if not use_mask_mc2:
        worker.determine_available_memory()
        worker.compile_or_warm_up_model()


def destroy_comm_group(use_mask_mc2: bool) -> None:
    if use_mask_mc2:
        get_dp_group().destroy_cpu_group()
        if get_ascend_config().eplb_config.dynamic_eplb:
            get_dynamic_eplb_group().destroy_cpu_group()
    else:
        destroy_ascend_model_parallel()
        cleanup_dist_env_and_memory()


def init_dp_cpu_group(vllm_config: VllmConfig, group_type="normal") -> None:
    if get_ascend_config().eplb_config.dynamic_eplb:
        get_dynamic_eplb_group().cpu_group = stateless_init_torch_distributed_process_group(
            vllm_config.parallel_config.data_parallel_master_ip,
            vllm_config.parallel_config.data_parallel_master_port + 200,
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
            backend="gloo",
            gloo_timeout_seconds=vllm_config.parallel_config.fault_tolerance_config.gloo_comm_timeout,
        )
        get_dynamic_eplb_group().group_type = group_type

    # TODO: Temporarily hardcode the port value for debugging. Will replace with get_open_port().
    get_dp_group().cpu_group = stateless_init_torch_distributed_process_group(
        vllm_config.parallel_config.data_parallel_master_ip,
        vllm_config.parallel_config.data_parallel_master_port + 100,
        vllm_config.parallel_config.data_parallel_rank,
        vllm_config.parallel_config.data_parallel_size,
        backend="gloo",
    )
    get_dp_group().group_type = group_type
    timeout = timedelta(seconds=vllm_config.parallel_config.fault_tolerance_config.gloo_comm_timeout)
    _set_pg_timeout(timeout=timeout, group=get_dp_group().cpu_group)


def reinit_comm_group(use_mask_mc2: bool, vllm_config: VllmConfig, worker: NPUWorker, coord_store=None) -> None:
    if use_mask_mc2:
        init_dp_cpu_group(vllm_config, "stateless")
    else:
        worker._init_worker_distributed_environment()


def _is_mtp_speculative(vllm_config) -> bool:
    spec_config = getattr(vllm_config, "speculative_config", None)
    if spec_config is None:
        return False
    method = getattr(spec_config, "method", None)
    return method == "mtp" or (isinstance(method, str) and method.endswith("_mtp"))


def _get_mtp_num_layers(vllm_config) -> int:
    if not _is_mtp_speculative(vllm_config):
        return 0

    draft = getattr(vllm_config.speculative_config, "draft_model_config", None)
    if draft is not None:
        num = draft.get_total_num_hidden_layers()
        if num > 0:
            return num

    hf = vllm_config.model_config.hf_config
    for attr in ("num_nextn_predict_layers", "mtp_num_hidden_layers", "n_predict"):
        num = getattr(hf, attr, None)
        if num:
            return num

    raise RuntimeError(
        "MTP layer count not found in model config; unsupported model configuration."
    )


def save_expert_weights_to_ram(
    cur_rank_need_load_h2d,
    vllm_config,
    model_runner,
    quant,
) -> dict[str, torch.Tensor]:
    """
    Load the specified unsaved expert weights and save them to memory (RAM)

    Args:
        cur_rank_need_load_h2d: Experts to be loaded from SSD
        vllm_config: VLLM configuration object
        model_runner: NPU model runner
        quant: Whether it is a quantized model (affects the suffix of weight names)

    Returns:
        updated dictionary of saved expert weights
    """

    BASE_WEIGHT_SUFFIXES = {"down_proj.weight", "up_proj.weight", "gate_proj.weight"}
    QUANT_WEIGHT_SUFFIXES = {
        "down_proj.weight_offset",
        "up_proj.weight_offset",
        "gate_proj.weight_offset",
        "down_proj.weight_scale",
        "up_proj.weight_scale",
        "gate_proj.weight_scale",
    }

    weight_suffixes = BASE_WEIGHT_SUFFIXES.union(QUANT_WEIGHT_SUFFIXES) if quant else BASE_WEIGHT_SUFFIXES

    def _generate_expert_weight_name(layer_id: int, expert_id: int, suffix: str) -> str:
        if layer_id < num_hidden_layers:
            return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{suffix}"

        mtp_local_idx = layer_id - num_hidden_layers
        if mtp_local_idx >= num_mtp_layers:
            raise RuntimeError(
                f"MTP layer index out of range: layer_id={layer_id}, "
                f"mtp_local_idx={mtp_local_idx}, num_mtp_layers={num_mtp_layers}"
            )

        model_type = getattr(vllm_config.model_config.hf_config, "model_type", "")
        if model_type in ("nemotron_h", "nemotron_h_mtp"):
            return f"mtp.layers.{mtp_local_idx}.mlp.experts.{expert_id}.{suffix}"
        if model_type in ("qwen3_next", "qwen3_next_mtp",
                           "qwen3_5", "qwen3_5_moe", "qwen3_5_mtp"):
            return f"mtp.layers.{mtp_local_idx}.mlp.experts.{expert_id}.{suffix}"
        if model_type in ("exaone_moe", "exaone_moe_mtp"):
            return f"mtp.layers.{mtp_local_idx}.mlp.experts.{expert_id}.{suffix}"
        if model_type in ("glm_moe_dsa",):
            # Checkpoint原始权重名不含mtp_block，vllm的_rewrite_spec_layer_name
            # 会在load_weights时自动加上mtp_block，所以这里要和checkpoint原始名保持一致
            return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{suffix}"
        if model_type in ("longcat_flash", "longcat_flash_mtp"):
            return (
                f"model.mtp.layers.{mtp_local_idx}.transformer_layer.mlp.experts."
                f"{expert_id}.{suffix}"
            )
        if model_type in ("ernie4_5_moe", "ernie_mtp"):
            return f"model.mtp_block.0.mlp.experts.{expert_id}.{suffix}"

        return f"model.layers.{layer_id}.mlp.experts.{expert_id}.{suffix}"

    num_dense_layers = getattr(model_runner.model.config, "first_k_dense_replace", 0)
    num_hidden_layers = getattr(model_runner.model.config, "num_hidden_layers", 0)
    num_mtp_layers = _get_mtp_num_layers(vllm_config)
    num_main_moe_layers = num_hidden_layers - num_dense_layers

    weights_to_save = set()

    # Main model layers
    for index, cur_layer_need_load_h2d in enumerate(cur_rank_need_load_h2d[:num_main_moe_layers]):
        layer_id = index + num_dense_layers
        if cur_layer_need_load_h2d:
            for pos, expert_id in cur_layer_need_load_h2d:
                for suffix in weight_suffixes:
                    weights_to_save.add(_generate_expert_weight_name(layer_id, expert_id, suffix))

    # MTP layers
    if num_mtp_layers > 0:
        for mtp_idx in range(num_mtp_layers):
            layer_id = num_hidden_layers + mtp_idx
            data_idx = num_main_moe_layers + mtp_idx
            if data_idx < len(cur_rank_need_load_h2d) and cur_rank_need_load_h2d[data_idx]:
                for pos, expert_id in cur_rank_need_load_h2d[data_idx]:
                    for suffix in weight_suffixes:
                        weights_to_save.add(_generate_expert_weight_name(layer_id, expert_id, suffix))

    model_loader = get_model_loader(vllm_config.load_config)

    # Load from main model
    all_weight_iter = model_loader.get_all_weights(vllm_config.model_config, model_runner.model)
    saved_expert_weights = {}
    for weight_name, weight_tensor in all_weight_iter:
        if weight_name in weights_to_save:
            weight_tensor = weight_tensor.transpose(0, 1).contiguous()
            if any(weight_name.endswith(suffix) for suffix in QUANT_WEIGHT_SUFFIXES):
                weight_tensor = torch.squeeze(weight_tensor)
            saved_expert_weights[weight_name] = weight_tensor

    # Load from draft model
    drafter = getattr(model_runner, "drafter", None)
    if drafter is not None and hasattr(drafter, "model") and num_mtp_layers > 0:
        draft_weight_iter = model_loader.get_all_weights(vllm_config.model_config, drafter.model)
        for weight_name, weight_tensor in draft_weight_iter:
            if weight_name in weights_to_save and weight_name not in saved_expert_weights:
                weight_tensor = weight_tensor.transpose(0, 1).contiguous()
                if any(weight_name.endswith(suffix) for suffix in QUANT_WEIGHT_SUFFIXES):
                    weight_tensor = torch.squeeze(weight_tensor)
                saved_expert_weights[weight_name] = weight_tensor

    return saved_expert_weights


def expand_parameter(old_param, axis: int = 0, extra_lines: int = 1) -> torch.nn.Parameter:
    old_shape = old_param.shape
    new_shape = list(old_param.shape)
    new_shape[axis] += extra_lines
    new_tensor = torch.zeros(
        size=new_shape,
        device=old_param.device,
        dtype=old_param.dtype,
    )
    if axis == 0:
        new_tensor[: old_shape[0]] = old_param.data
    elif axis == 1:
        new_tensor[:, : old_shape[1]] = old_param.data
    else:
        raise NotImplementedError(f"axis {axis} is not supported")
    return torch.nn.Parameter(new_tensor, requires_grad=old_param.requires_grad)


def expand_expert_weights(model_runner: NPUModelRunner, expand_lines: int, quant: bool | str) -> None:
    if not expand_lines:
        return
    models = [model_runner.model]
    draft_model = getattr(getattr(model_runner, "drafter", None), "model", None)
    if draft_model is not None:
        models.append(draft_model)
    for model in models:
        for module in model.modules():
            if isinstance(module, FusedMoE):
                if quant:
                    module.w2_weight_list = expand_parameter(module.w2_weight_list, 0, expand_lines)
                    module.w13_weight_list = expand_parameter(module.w13_weight_list, 0, expand_lines)
                    module.w2_weight_scale_list = expand_parameter(module.w2_weight_scale_list, 0, expand_lines)
                    module.w13_weight_scale_fp32_list = expand_parameter(
                        module.w13_weight_scale_fp32_list, 0, expand_lines
                    )
                    module.w2_weight_offset.data = expand_parameter(module.w2_weight_offset.data, 0, expand_lines)
                    module.w13_weight_offset.data = expand_parameter(module.w13_weight_offset.data, 0, expand_lines)
                else:
                    module.w2_weight = expand_parameter(module.w2_weight, 0, expand_lines)
                    module.w13_weight = expand_parameter(module.w13_weight, 0, expand_lines)


def dynamic_merge_view(
    target_tensor: torch.Tensor, tensor1: torch.Tensor, tensor2: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    dim_size1 = tensor1.shape[dim]
    dim_size2 = tensor2.shape[dim]
    total_dim_size = dim_size1 + dim_size2

    non_dim_shapes = [s for i, s in enumerate(tensor1.shape) if i != dim]
    for i, s in enumerate(tensor2.shape):
        if i != dim and s != non_dim_shapes[i if i < dim else i - 1]:
            raise ValueError(f"size mismatch on non merged dimension {i}：tensor1={s} vs tensor2 = {non_dim_shapes[i]}")
    if target_tensor.shape[dim] != total_dim_size:
        raise ValueError(f"target tensor on dim {dim} must be {dim_size1}+{dim_size2}={total_dim_size}")

    top_view = target_tensor.narrow(dim, 0, dim_size1)
    bottom_view = target_tensor.narrow(dim, dim_size1, dim_size2)

    top_view.copy_(tensor1)
    bottom_view.copy_(tensor2)

    return target_tensor


def reload_fault_expert_weights(
    model_runner: NPUModelRunner,
    cur_rank_need_load_h2d,
    experts_saved_weights: dict[str, torch.Tensor],
    quant: bool | str = False,
) -> None:
    def _load_single_expert(expert_id: int, target_index: int, quant: bool | str = False):
        prefix = f"{module.layer_name}.{expert_id}"
        w1_key = f"{prefix}.gate_proj.weight"
        w2_key = f"{prefix}.down_proj.weight"
        w3_key = f"{prefix}.up_proj.weight"
        # GLM-5 MTP: checkpoint原始权重名不含mtp_block，但module.layer_name含mtp_block
        if w1_key not in experts_saved_weights and "mtp_block" in prefix:
            alt_prefix = prefix.replace(".mtp_block.mlp.", ".mlp.", 1)
            w1_key = f"{alt_prefix}.gate_proj.weight"
            w2_key = f"{alt_prefix}.down_proj.weight"
            w3_key = f"{alt_prefix}.up_proj.weight"
        elif w1_key not in experts_saved_weights and "mtp_block" not in prefix:
            alt_prefix = prefix.replace(".mlp.", ".mtp_block.mlp.", 1)
            w1_key = f"{alt_prefix}.gate_proj.weight"
            w2_key = f"{alt_prefix}.down_proj.weight"
            w3_key = f"{alt_prefix}.up_proj.weight"
        w1_weight = experts_saved_weights[w1_key]
        w2_weight = experts_saved_weights[w2_key]
        w3_weight = experts_saved_weights[w3_key]
        if get_ascend_config().eplb_config.dynamic_eplb:
            device = module.w2_weight_list[target_index].device
            module._load_w2(
                expert_data=module.w2_weight_list[target_index],
                shard_dim=1,
                loaded_weight=w2_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight_list[target_index],
                shard_dim=1,
                shard_id="w1",
                loaded_weight=w1_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight_list[target_index],
                shard_dim=1,
                shard_id="w3",
                loaded_weight=w3_weight.to(device),
                tp_rank=module.tp_rank,
            )
        else:
            device = module.w2_weight.device
            module._load_w2(
                expert_data=module.w2_weight[target_index],
                shard_dim=1,
                loaded_weight=w2_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight[target_index],
                shard_dim=1,
                shard_id="w1",
                loaded_weight=w1_weight.to(device),
                tp_rank=module.tp_rank,
            )
            module._load_w13(
                expert_data=module.w13_weight[target_index],
                shard_dim=1,
                shard_id="w3",
                loaded_weight=w3_weight.to(device),
                tp_rank=module.tp_rank,
            )

        if quant:
            s1_key = f"{prefix}.gate_proj.weight_scale"
            s2_key = f"{prefix}.down_proj.weight_scale"
            s3_key = f"{prefix}.up_proj.weight_scale"
            o1_key = f"{prefix}.gate_proj.weight_offset"
            o2_key = f"{prefix}.down_proj.weight_offset"
            o3_key = f"{prefix}.up_proj.weight_offset"
            if s1_key not in experts_saved_weights and "mtp_block" in prefix:
                alt_prefix = prefix.replace(".mtp_block.mlp.", ".mlp.", 1)
                s1_key = f"{alt_prefix}.gate_proj.weight_scale"
                s2_key = f"{alt_prefix}.down_proj.weight_scale"
                s3_key = f"{alt_prefix}.up_proj.weight_scale"
                o1_key = f"{alt_prefix}.gate_proj.weight_offset"
                o2_key = f"{alt_prefix}.down_proj.weight_offset"
                o3_key = f"{alt_prefix}.up_proj.weight_offset"
            elif s1_key not in experts_saved_weights and "mtp_block" not in prefix:
                alt_prefix = prefix.replace(".mlp.", ".mtp_block.mlp.", 1)
                s1_key = f"{alt_prefix}.gate_proj.weight_scale"
                s2_key = f"{alt_prefix}.down_proj.weight_scale"
                s3_key = f"{alt_prefix}.up_proj.weight_scale"
                o1_key = f"{alt_prefix}.gate_proj.weight_offset"
                o2_key = f"{alt_prefix}.down_proj.weight_offset"
                o3_key = f"{alt_prefix}.up_proj.weight_offset"
            w1_weight_scale = experts_saved_weights[s1_key].to(device)
            w2_weight_scale = experts_saved_weights[s2_key].to(device)
            w3_weight_scale = experts_saved_weights[s3_key].to(device)
            w1_weight_offset = experts_saved_weights[o1_key].to(device)
            w2_weight_offset = experts_saved_weights[o2_key].to(device)
            w3_weight_offset = experts_saved_weights[o3_key].to(device)
            module.w2_weight_offset.data[target_index].copy_(w2_weight_offset)
            dynamic_merge_view(module.w13_weight_offset.data[target_index], w1_weight_offset, w3_weight_offset)
            if get_ascend_config().eplb_config.dynamic_eplb:
                module.w2_weight_scale_list[target_index].copy_(w2_weight_scale)
                dynamic_merge_view(module.w13_weight_scale_fp32_list[target_index], w1_weight_scale, w3_weight_scale)
            else:
                module.w2_weight_scale[target_index].copy_(w2_weight_scale)
                dynamic_merge_view(module.w13_weight_scale_fp32[target_index], w1_weight_scale, w3_weight_scale)

    cur_layer_id = 0
    for module in model_runner.model.modules():
        if isinstance(module, FusedMoE):
            if cur_rank_need_load_h2d[cur_layer_id] is not None:
                for slot_pos, expert_id in cur_rank_need_load_h2d[cur_layer_id]:
                    _load_single_expert(expert_id=expert_id, target_index=slot_pos, quant=quant)
            cur_layer_id += 1

    draft_model = getattr(getattr(model_runner, "drafter", None), "model", None)
    if draft_model is not None:
        for module in draft_model.modules():
            if isinstance(module, FusedMoE):
                if cur_rank_need_load_h2d[cur_layer_id] is not None:
                    for slot_pos, expert_id in cur_rank_need_load_h2d[cur_layer_id]:
                        _load_single_expert(expert_id=expert_id, target_index=slot_pos, quant=quant)
                cur_layer_id += 1


def update_parallel_config(original_config: VllmConfig, update_config: dict[str, int]) -> None:  # , worker_guard)
    required_keys = {
        "data_parallel_size",
        "data_parallel_size_local",
        "data_parallel_rank",
        "data_parallel_master_port",
    }
    missing_keys = required_keys - set(update_config.keys())
    if missing_keys:
        raise ValueError(f"update parallel config failed missing keys: {missing_keys}")

    original_config.parallel_config.data_parallel_size = update_config["data_parallel_size"]
    original_config.parallel_config.data_parallel_size_local = update_config["data_parallel_size_local"]
    original_config.parallel_config.data_parallel_rank = update_config["data_parallel_rank"]
    original_config.parallel_config.data_parallel_master_port = update_config["data_parallel_master_port"]


def init_ep2dp_map(dp_size: int, tp_size: int) -> dict[int, int]:
    ep2dp_map = {}
    for dp_rank in range(dp_size):
        ep_start = dp_rank * tp_size
        ep_end = (dp_rank + 1) * tp_size
        for ep_rank in range(ep_start, ep_end):
            ep2dp_map[ep_rank] = dp_rank
    return ep2dp_map


def update_ep2dp_map(
    ep2dp_map: dict[int, int],
    exclude_dp_ranks: list[int],
    rank_mapping: dict[int, int],
) -> dict[int, int]:
    for old_ep_rank, dp_rank in ep2dp_map.items():
        if dp_rank != -1:
            if dp_rank in exclude_dp_ranks:
                ep2dp_map[old_ep_rank] = -1
            else:
                ep2dp_map[old_ep_rank] = rank_mapping[dp_rank]
    return ep2dp_map


def init_elastic_info(
    use_mask_mc2: bool,
    ep_size: int,
    phy_experts_num: int,
    share_expert_rank_num: int = 0,
):
    if use_mask_mc2:
        # ----- 1 Basic configuration (first 4 parameters) -----
        # Meaning: whether to descale (0 = no descale), actual number of ranks after descale
        # reduction (=ep_size), number of ranks for shared experts,number of MoE experts
        descale = 0
        base_config = torch.tensor([descale, ep_size, share_expert_rank_num, phy_experts_num], dtype=torch.int32)

        # ----- 2 Mapping tables -----
        # Table1: epRankID -> localEpRankId(-1 indicates invalid）
        table1 = torch.arange(0, ep_size, dtype=torch.int32)
        # Table2: localEpRankId -> epRankID(-1 indicates padding）
        table2 = torch.arange(0, ep_size, dtype=torch.int32)

        # ----- 3 Concatenate into a complete 1D Tensor -----
        elastic_info = torch.cat([base_config, table1, table2], dim=0).npu().contiguous()

        # ---- 4 Configure Tensor properties and set global variables
        elastic_info.requires_grad_(False)
        set_elastic_info(elastic_info)


def update_elastic_info(
    use_mask_mc2: bool,
    elastic_info: torch.Tensor,
    expert_num: int,
    raw_ep_size: int,
    ep2dp: dict[int, int],
    share_expert_num: int = 0,
) -> None:
    if use_mask_mc2:
        if elastic_info is None:
            elastic_info = torch.full((4 + 2 * raw_ep_size,), -1, dtype=torch.int32).npu().contiguous()
        raw_ep_ranks = sorted(ep2dp.keys())
        valid_ep_ranks = [ep for ep in raw_ep_ranks if ep2dp[ep] != -1]
        descale_ep_size = len(valid_ep_ranks)
        is_descale = 1 if descale_ep_size < raw_ep_size else 0

        # Table1: epRankID -> localEpRankId(-1 indicates invalid）
        table1 = torch.full((raw_ep_size,), -1, dtype=torch.int32, device="cpu")
        for local_ep_rank, ep_rank in enumerate(valid_ep_ranks):
            table1[ep_rank] = local_ep_rank

        # Table2: localEpRankId -> epRankID(-1 indicates padding）
        table2 = torch.full((raw_ep_size,), -1, dtype=torch.int32, device="cpu")
        for local_ep_rank, ep_rank in enumerate(valid_ep_ranks):
            if local_ep_rank < descale_ep_size:
                table2[local_ep_rank] = ep_rank

        # update elastic_info
        elastic_info[0] = is_descale
        elastic_info[1] = descale_ep_size
        elastic_info[2] = share_expert_num
        elastic_info[3] = expert_num
        # update Table1
        table1_start = 4
        elastic_info[table1_start : table1_start + raw_ep_size] = table1
        # update Table2
        table2_start = table1_start + raw_ep_size
        elastic_info[table2_start : table2_start + raw_ep_size] = table2
        set_elastic_info(elastic_info)
    else:
        set_elastic_info(None)


def gen_local_log2phy_map(global_log2phy_map: dict[int, list[int]]) -> torch.Tensor:
    num_logical_exp = len(global_log2phy_map)
    log2phy = torch.zeros(num_logical_exp, dtype=torch.int32, device="cpu")
    for log_expert_id in sorted(global_log2phy_map.keys()):
        replica_list = global_log2phy_map[log_expert_id]
        # num_replicas = len(replica_list)
        # phy_id = replica_list[global_rank % num_replicas]
        # TODO: For now we can only use the 0-th physical expert of each logical expert;
        # using the two lines above for load balancing causes accuracy issues.
        phy_id = replica_list[0]
        log2phy[log_expert_id] = phy_id
    return log2phy.npu()


def reconfigure_moe(
    use_mask_mc2: bool,
    modelrunner: NPUModelRunner,
    vllm_config: VllmConfig,
    num_global_logical_experts: int,
    num_global_new_phy_experts: int,
    log2phy: torch.Tensor,
):
    import vllm.envs as envs

    parallel_config = vllm_config.parallel_config
    new_ep_size = parallel_config.data_parallel_size * parallel_config.tensor_parallel_size
    get_ascend_config().eplb_config.num_redundant_experts = num_global_new_phy_experts - num_global_logical_experts

    moe_modules = [module for module in modelrunner.model.modules() if isinstance(module, FusedMoE)]
    draft_model = getattr(getattr(modelrunner, "drafter", None), "model", None)
    if draft_model is not None:
        moe_modules.extend(
            module for module in draft_model.modules() if isinstance(module, FusedMoE)
        )

    for cur_layer_id, module in enumerate(moe_modules):
        module.local_num_experts = num_global_new_phy_experts // new_ep_size
        module.global_num_experts = num_global_new_phy_experts
        module.global_redundant_expert_num = num_global_new_phy_experts - num_global_logical_experts
        sp_size = module.sp_size
        module.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size_=get_tp_group().world_size,
            pcp_size_=get_pcp_group().world_size,
            dp_size_=get_dp_group().world_size,
            vllm_parallel_config=parallel_config,
            sp_size_=sp_size,
        )
        module.moe_config = FusedMoEConfig(
            num_experts=module.global_num_experts,
            experts_per_token=module.top_k,
            hidden_dim=module.hidden_size,
            intermediate_size_per_partition=module.intermediate_size_per_partition,
            num_local_experts=module.local_num_experts,
            num_logical_experts=num_global_logical_experts,
            moe_parallel_config=module.moe_parallel_config,
            in_dtype=module.vllm_config.model_config.dtype,
            router_logits_dtype=None,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=False,
            is_act_and_mul=True,
            is_lora_enabled=module.vllm_config.lora_config is not None,
            activation=module.activation,
            device=module.vllm_config.device_config.device,
            routing_method=module.routing_method_type,
        )
        module.moe_config.num_experts = module.global_num_experts
        module.moe_config.num_local_experts = module.local_num_experts
        module.moe_config.global_redundant_expert_num = module.global_redundant_expert_num
        module.log2phy.copy_(log2phy[cur_layer_id].npu(), non_blocking=True)
        if not use_mask_mc2:
            module.moe_config.tp_group = get_tp_group()
            module.moe_config.dp_group = get_dp_group()
            module.moe_config.ep_group = get_ep_group()
            module.moe_config.mc2_group = get_mc2_group()
            module.moe_config.supports_eplb = module.quant_method.supports_eplb
            eplb_config = get_ascend_config().eplb_config
            module.global_expert_map, module._expert_map, _, _ = init_eplb_config(
                eplb_config, module.moe_counter, module.moe_config
            )
            with set_current_vllm_config(vllm_config):
                setup_moe_comm_method(module.moe_config)
            changed_moe_load_shape = module.local_num_experts - module.moe_load.shape[0]
            if modelrunner.dynamic_eplb and changed_moe_load_shape:
                module.moe_load = expand_parameter(module.moe_load, 0, changed_moe_load_shape)
            if vllm_config.model_config.quantization is not None:
                from vllm_ascend.quantization.methods.w8a8_dynamic import AscendW8A8DynamicFusedMoEMethod

                module.quant_method.quant_method = AscendW8A8DynamicFusedMoEMethod()
                # todo support other quant like w4a4 w4a8 ...


def update_eplb_adaptor_info(model_runner, num_add_experts_per_rank, rank):
    model_runner.eplb_adaptor.rank_id = rank
    model_runner.eplb_adaptor.model.clear_all_moe_loads()
    model_runner.shared_dict["moe_load"] = None
    model_runner.eplb_updator.cur_iterations = 0

    if num_add_experts_per_rank > 0:
        model_runner.eplb_adaptor.init_buffer_tensor(num_add_experts_per_rank)

    model_runner.eplb_adaptor.init_expert_param_per_layer()
    cur_deployment = model_runner.shared_dict["expert_maps"]
    for layer_id in range(cur_deployment.shape[0]):
        model_runner.eplb_adaptor.do_clone_update_expert_map(layer_id, cur_deployment[layer_id][rank])


def d2d_transmission_for_scaling_down(model_runner):
    eplb_loader = model_runner.eplb_loader
    eplb_adaptor = model_runner.eplb_adaptor
    eplb_updator = model_runner.eplb_updator

    all_layer_log2phy_map = []

    while eplb_updator.update_info_all:
        (expert_send_info, expert_recv_info, updated_expert_map, log2phy_map, layer_id) = (
            eplb_updator.update_info_all.pop(0)
        )

        log2phy_map_this_rank = torch.from_numpy(np.array(log2phy_map))
        all_layer_log2phy_map.append(log2phy_map_this_rank)
        eplb_loader.set_log2phy_map(log2phy_map_this_rank)
        updated_expert_map_this_rank = torch.from_numpy(np.array(updated_expert_map))

        eplb_loader.generate_expert_d2d_transfer_task(
            expert_send_info,
            expert_recv_info,
            updated_expert_map_this_rank,
            layer_id + eplb_adaptor.num_dense_layers,
        )

        reqs = []
        eplb_loader.asyn_expert_weight_transfer(reqs)
        eplb_loader.update_expert_map_and_weight(reqs)

    # Append for MTP layers (same log2phy as corresponding main model layers)
    vllm_config = getattr(model_runner, "vllm_config", None)
    num_mtp_layers = _get_mtp_num_layers(vllm_config) if vllm_config is not None else 0
    if num_mtp_layers > 0:
        num_main_moe_layers = len(all_layer_log2phy_map)
        for mtp_layer_idx in range(num_mtp_layers):
            main_layer_idx = mtp_layer_idx % num_main_moe_layers if num_main_moe_layers > 0 else 0
            all_layer_log2phy_map.append(all_layer_log2phy_map[main_layer_idx].clone())

    torch_npu.npu.synchronize()

    return all_layer_log2phy_map


def gen_all_layer_log2phy(model_runner, rank):
    all_layer_log2phy = []
    cur_deployment = model_runner.shared_dict["expert_maps"]
    for layer_id in range(cur_deployment.shape[0]):
        cur_layer_log2phy_map = generate_log2phy_map(cur_deployment[layer_id], rank)
        all_layer_log2phy.append(cur_layer_log2phy_map)

    # Append for MTP layers (same log2phy as corresponding main model layers)
    vllm_config = getattr(model_runner, "vllm_config", None)
    num_mtp_layers = _get_mtp_num_layers(vllm_config) if vllm_config is not None else 0
    if num_mtp_layers > 0:
        num_main_moe_layers = len(all_layer_log2phy)
        for mtp_layer_idx in range(num_mtp_layers):
            main_layer_idx = mtp_layer_idx % num_main_moe_layers if num_main_moe_layers > 0 else 0
            all_layer_log2phy.append(all_layer_log2phy[main_layer_idx].clone())

    return all_layer_log2phy


def get_global_expert_map(model_runner):
    num_dense_layers = getattr(model_runner.model.config, "first_k_dense_replace", 0)
    num_moe_layers = model_runner.model.config.num_hidden_layers - num_dense_layers
    all_layer_global_expert_map = []
    for layer_id in range(num_moe_layers):
        map_cpu = model_runner.model.model.layers[num_dense_layers + layer_id].mlp.experts.global_expert_map.cpu()
        all_layer_global_expert_map.append(map_cpu)
    vllm_config = getattr(model_runner, "vllm_config", None)
    num_mtp_layers = _get_mtp_num_layers(vllm_config)
    if num_mtp_layers > 0:
        drafter = getattr(model_runner, "drafter", None)
        if drafter is not None and hasattr(drafter, "model"):
            mtp_model = drafter.model
            for mtp_layer_idx in range(num_mtp_layers):
                mtp_layer = mtp_model.model.layers[mtp_layer_idx]
                map_cpu = mtp_layer.mlp.experts.global_expert_map.cpu()
                all_layer_global_expert_map.append(map_cpu)

    return torch.stack(all_layer_global_expert_map)
