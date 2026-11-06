"""
engine/misc/profiler_utils.py
"""

import copy
from calflops import calculate_flops
from typing import Tuple

def stats(
    cfg,
    input_shape: Tuple = (1, 3, 512, 640),
) -> Tuple[int, dict]:
    r"""计算模型 FLOPs 和参数量，彻底支持 [H=512, W=640] 矩形分辨率."""

    # 1. 深度穿透读取真实配置尺寸 (优先查 yaml_cfg 字典)
    eval_size = None
    if hasattr(cfg, "yaml_cfg") and isinstance(cfg.yaml_cfg, dict) and "eval_spatial_size" in cfg.yaml_cfg:
        eval_size = cfg.yaml_cfg["eval_spatial_size"]
    elif hasattr(cfg, "eval_spatial_size") and cfg.eval_spatial_size is not None:
        eval_size = cfg.eval_spatial_size

    # 2. 精确构造假输入尺寸: (1, 3, H, W)
    if eval_size is not None:
        input_shape = (1, 3, int(eval_size[0]), int(eval_size[1]))
    else:
        base_size = cfg.train_dataloader.collate_fn.base_size
        input_shape = (1, 3, base_size, base_size)

    # 3. 统计部署态 FLOPs
    # deploy 将模型置 eval 并执行可重参数化转换，统计的是推理路径而非训练双流。
    # 三通道 dummy 会由双流主干复制为六通道；这只用于算量，不检验模态真实性。
    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )
    # InfraNet 的 IR 模式只是旁路 RGB，不删除辅助模块，所以此总参数量
    # 仍包含未执行的 RGB/QualGate 参数，不能直接当作裁剪后的部署体积。
    # calflops 对 grid_sample/einsum 等算子的计数口径需确认，FLOPs 不是实测 FPS。
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
