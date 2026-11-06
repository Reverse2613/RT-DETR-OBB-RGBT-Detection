# RT-DETR-OBB-RGBT-Detection：代码导读、复现指南与创新审查

这是一个面向 **RGB + 热红外（TIR/IR）旋转目标检测** 的研究代码仓库。它使用 HGNetv2 等骨干、混合特征编码器和 RT-DETR 风格的旋转框解码器，包含 ATR-UMOD、DroneVehicle 数据接口，以及若干多模态融合实验。

本 README 以**当前工作区实际存在的代码**为依据，覆盖 **97 个 Python 文件和 14 个 YAML 配置文件**。既说明当前使用的主链路，也区分保留的通用模块、辅助脚本和未完成实验，不能把所有文件都视为已经验证可用的功能。

> **本次修改范围：仅编写文档、给主要 Python 文件增加中文注释，不修改可执行逻辑、接口、配置、权重或原有 docstring。** 下文指出的问题仍保留在代码中，不能误以为已经修复。审查包含静态分析和独立的小张量验证，没有重新训练模型、验证完整数据集或核实外部论文。

## 目录

- [1. 项目定位与完成程度](#overview)
- [2. 整体调用链与张量约定](#pipeline)
- [3. 环境、训练、恢复与验证](#usage)
- [4. 数据集与预处理](#datasets)
- [5. 全部配置文件说明](#configs)
- [6. 全部 Python 文件逐一说明](#file-atlas)
- [7. 主要代码的详细解释](#core-details)
- [8. 代码审查：问题、证据与风险](#audit)
- [9. 已有实验记录如何解读](#existing-results)
- [10. 能否创新：理论与代码实验路线](#research)
- [11. 验证、复现与后续工作清单](#verification)
- [12. 许可证与说明边界](#license)

<a id="overview"></a>
## 1. 项目定位与完成程度

### 1.1 它解决什么问题

输入同一场景的可见光图像和热红外图像，输出目标的类别、置信度及旋转框。旋转框适合车辆等方向性明显的遥感目标，比水平框能更紧密地描述目标占据的区域。

- **RGB** 通常提供纹理、颜色和细节，但容易受黑夜、眩光、天气影响。
- **IR** 通常提供不同于 RGB 的辐射/热对比信息，但也可能缺少纹理或受热交叉、噪声影响。
- **融合** 的目标不是无条件增加 RGB 权重，而是在配准、质量、场景和算力约束下利用互补信息。
- **OBB** 表示旋转框；代码内部并非所有阶段都使用相同的角度单位，必须遵守第 2 节的约定。

仓库的工程组织沿用 RT-DETR/RT-DETRv4 风格。`experiment.md` 描述了从其他实验工程抽取代码的背景；这里不根据文件名或头部声明断言某篇论文已经被完整复现。

### 1.2 当前可以从代码确认的实验分支

| 分支 | 主要实现 | 当前作用 | 审查结论 |
| --- | --- | --- | --- |
| 单模态 IR | `SingleHGNetv2_IR` | 只提取输入前三个 IR 通道，作为单模态对照 | 有完整配置；当前数据加载仍采用成对数据集 |
| 双流相加 | `DualHGNetv2_Add` | 两套 HGNetv2 的同尺度特征直接相加 | 简单融合基线，适合做后续模块的主要对照 |
| 双流拼接 | `DualHGNetv2_Concat` | 同尺度拼接后，用 1×1 卷积压回原通道数 | 有额外融合参数，不能只按融合方式比较而忽略参数量 |
| 条件监督融合 | `DualHGNetv2_LGDense`，prototype 配置 | 用联合条件向量控制全局通道门和 RGB 局部门，并监督六组条件 | 已实现；条件并不是六维隐向量，也不是经过物理校准的质量量 |
| 无显式条件监督融合 | 同一骨干，self-driven 配置 | 不创建六组辅助预测头，保留条件编码器和融合门 | 消融的是辅助监督，不是整个条件感知/门控结构 |
| InfraNet 风格分支 | `DualHGNetv2_InfraNet`、`RTv4_Infranet` | 完整 IR 主干 + 轻量 RGB 分支 + 质量门；训练同时计算主/辅助检测 | 有 IR-only 与 RGB-IR 推理路径；当前 YAML 实际选择 RGB-IR |
| TSTI/Sup fast | DroneVehicle TSTI YAML | 声明非对称骨干和显著性损失 | **缺少对应骨干实现，损失映射也不完整，当前不能直接运行** |

**“有完整配置”只表示组件链条存在，不表示本次已经完成端到端运行验证。** 依赖安装、数据路径、标注格式、预训练文件和第 8 节中的问题仍需检查。

### 1.3 建议的阅读顺序

1. 先读一个实验 YAML，例如 `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml`。
2. 再读 `train.py`、`engine/core/yaml_config.py`、`engine/core/workspace.py`，理解对象是如何由配置构建的。
3. 读 `atrumod_dataset.py`、`transforms_rgbt.py`、`dataloader.py`，确认模态顺序与框的单位。
4. 沿 `RTv4 → backbone → HybridEncoder → RTDETRTransformerv2OBB` 阅读模型。
5. 最后读 `obb_matcher.py`、`rtv4_obb_criterion.py`、`postprocessor.py`、`dota_evaluator.py`，理解训练目标和报告指标。

<a id="pipeline"></a>
## 2. 整体调用链与张量约定

### 2.1 从配置到指标的主链路

```text
train.py
  ├─ 初始化设备、随机种子与分布式环境
  ├─ YAMLConfig：读取 include、合并覆盖参数
  │    └─ register/create：按注册表构建各个组件
  └─ TASKS['detection'] → DetSolver
       ├─ Dataset → 成对变换 Compose → BatchImageCollateFunction
       ├─ 模型 RTv4 / RTv4_Infranet
       │    ├─ HGNetv2 单流/双流骨干：S3、S4、S5
       │    ├─ HybridEncoder：投影、尺度内注意力、FPN/PAN
       │    └─ RTDETRTransformerv2OBB：查询选择、旋转可变形注意力、框预测
       ├─ 训练：HungarianMatcherOBB → RTv4OBBCriterion → 反向传播
       │         AMP / EMA / 优化器 / 学习率调度 / checkpoint
       └─ 验证：PostProcessorOBB → DotaEvaluator → mmeval → AP
```

训练时把 `targets` 传给模型，是为了构造去噪查询等训练专用信息。正常验证不应该向模型提供真值框；评估器单独读取真值进行评分。

`engine/__init__.py` 会导入子包以完成注册。**没有被当前模型选择的组件，也可能因模块级导入而影响启动依赖。** 因此“只运行 OBB”不等于可以省去所有 COCO、timm 或绘图库依赖。

### 2.2 RGB/IR 的顺序与尺寸

| 阶段 | 数据形态 | 必须注意的约定 |
| --- | --- | --- |
| 数据集读图 | `{'ir': PIL.Image, 'rgb': PIL.Image}` | 两幅图先转成 RGB 格式的 PIL 图像；灰度 IR 的三通道不代表三种独立热波段 |
| 成对图像变换 | 图像字典 + target | 裁剪、翻转、Mosaic 等必须对两个模态采用相同的几何参数 |
| 转 Tensor 后 | `[6, H, W]` | **前 3 通道是 IR，后 3 通道是 RGB**；不是名字 RGBT 所暗示的 RGB 在前 |
| 批处理后 | `[B, 6, H, W]` | `PairedConvertPILToTensor` 的 uint8→float 处理按配置执行；当前流水线转为 0～1 |
| HGNetv2-B0 输出 | 三个特征 Tensor | `return_idx=[1,2,3]`，通道 `[256,512,1024]`，步长 `[8,16,32]` |
| 编码器输出 | 三个投影/融合特征 Tensor | 当前主要配置统一为 224 通道 |
| 普通解码输出 | `pred_logits`、`pred_boxes` | 通常分别为 `[B,300,K]`、`[B,300,5]`；训练还包含动态数量的去噪查询与辅助输出 |

对于 ATR 配置的 `[H,W]=[512,640]`，三个尺度的空间尺寸分别约为 `[64,80]`、`[32,40]`、`[16,20]`。HGNetv2 的更早阶段 S2 是 64 通道、步长 4，不在当前三尺度输出中。

### 2.3 旋转框的单位变化

| 阶段 | 五维框内容 | 角度单位 |
| --- | --- | --- |
| 数据集产生的原始 target | `[cx_px, cy_px, w_px, h_px, angle_deg/180]` | 归一化角度，等价于 `theta/pi` |
| `PairedConvertOBB(normalize=True)` 后 | `[cx/W, cy/H, w/W, h/H, theta/pi]` | 仍为归一化角度 |
| 解码器预测 | 同上，五个量通过 sigmoid 等方式约束 | 角度预测分量在归一化域 |
| `PostProcessorOBB` 输出 | `[cx_px, cy_px, w_px, h_px, theta_rad]` | **弧度** |
| 本地旋转评估 | 像素框或四点多边形 | 旋转框输入按弧度解释 |

原始旋转框使用长边约定，通常先令 `w >= h`、角度归一到半周范围。**按 W/H 分别归一化后，数值上的 `w/W >= h/H` 不一定成立**；也不能直接把归一化坐标系当成等距的像素坐标系旋转。

OBB 数据路径中的 `orig_size` 是 `[H,W]`。仓库保留的部分 COCO 路径使用历史 `[W,H]` 约定，不能不经检查直接共用后处理。

### 2.4 六组环境条件

ATR 的 `target['conditions']` 是字典，描述飞行/环境条件，而不是每个目标的六维旋转框属性：

| 条件组 | 监督表示 | 当前归一化/类别范围 |
| --- | --- | --- |
| `height` | 单值回归 | `clip((UAVheight-80)/220, 0, 1)` |
| `angle` | 单值回归 | `clip(UAVangle/75, 0, 1)`；这是飞行观测角，不是目标 OBB 角度 |
| `time` | 两值回归 | `sin(2πt/24)`、`cos(2πt/24)`，避免午夜处的时间断点 |
| `weather` | 分类 | 10 类，索引 0～9 |
| `illumination` | 分类 | 7 类，索引 0～6 |
| `location` | 分类 | 11 类，索引 0～10 |

辅助头一共输出 `1+1+2+10+7+11=32` 个数；联合隐向量 `z_cond` 在 prototype 中是 256 维。条件缺失时目前会使用默认值并继续参与监督，**没有缺失标签掩码**，这是研究设计中的一个重要混杂因素。

<a id="usage"></a>
## 3. 环境、训练、恢复与验证

### 3.1 环境建议与依赖文件问题

建议使用独立环境，Python 3.10/3.11 起步，并按设备 CUDA 版本安装**相互匹配的 PyTorch 与 torchvision**。这只是起步建议，不是本项目的完整、经过测试的版本锁定。

`requirements.txt` 中 `torch==2.2.2`、`torchvision==0.17.2` 是被注释的历史提示，没有实际固定版本。不同 timm、Numba、NumPy 版本也可能带来额外兼容要求。

当前依赖文件有三个不能按标准 requirement 解析的条目：

```text
opencv-python#
transformers<4.45#
matplotlib#
```

`#` 前没有空白，不能当作正常行尾注释。因此**不要直接假设 `pip install -r requirements.txt` 会成功**。本次没有修改该文件；可以自行使用正确形式安装：

```powershell
python -m pip install torch torchvision
python -m pip install "faster-coco-eval>=1.6.5" PyYAML tensorboard scipy calflops opencv-python "timm>=1.0.27" matplotlib "numba>=0.65.1" "pycocotools==2.0.11" numpy Pillow packaging sympy
```

第一行需要根据 PyTorch 官方的 CPU/CUDA 安装方式调整，不能保证默认 wheel 适合你的显卡。`transformers<4.45` 在当前源文件中没有直接导入；如果其他工作流程确实需要它，再单独安装 `python -m pip install "transformers<4.45"`。不启用远程 DOTA 自动评估时，不需要为本地 AP 评估安装其远程客户端。

### 3.2 预训练权重

当前工作区有本地权重：

```text
pretrain/hgnetv2/PPHGNetV2_B0_stage1.pth
```

HGNetv2-B0 的主要配置使用 `local_model_dir: ./pretrain/hgnetv2/`。预训练目录被本地 `.gitignore` 忽略，因此**克隆代码并不保证得到该文件**。

缺少权重时，应明确选择：提供正确本地文件，或者通过 `-u 对应骨干.pretrained=False` 从零初始化。不要把“预训练开启但实际加载失败”和“从零训练”混在同一组比较中。当前 HGNetv2 下载/报错分支还存在未初始化分布式环境的风险，见第 8 节。

### 3.3 Windows 单卡训练示例

在仓库根目录执行。下面假设你已经建立 `D:/datasets/ATR-UMOD/train` 和独立的 `val` 划分；`val` 不是本仓库自带的数据，不应把官方 test 当作调参验证集后再报告独立测试成绩。

```powershell
$env:CUDA_VISIBLE_DEVICES = "0"
python train.py `
  -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml `
  --use-amp --seed 0 --output-dir outputs/atrumod_add_seed0 `
  -u "train_dataloader.dataset.img_folder=D:/datasets/ATR-UMOD/train/images_ir" `
     "train_dataloader.dataset.ann_folder=D:/datasets/ATR-UMOD/train/labels_ir" `
     "val_dataloader.dataset.img_folder=D:/datasets/ATR-UMOD/val/images_ir" `
     "val_dataloader.dataset.ann_folder=D:/datasets/ATR-UMOD/val/labels_ir" `
     "train_dataloader.num_workers=0" "val_dataloader.num_workers=0"
```

PowerShell 的反引号是续行符，行末不能再有空白。初次定位数据问题时 `num_workers=0` 更容易看到完整报错，确认正常后再根据硬件提高 worker 数量。IR/RGB 目录不能自动匹配时，可额外覆盖 `train_dataloader.dataset.img_folder_rgb`、`val_dataloader.dataset.img_folder_rgb`。

更换 `-c` 即可选择 single IR、concat、prototype、self-driven 或 InfraNet 配置；**仍需要传入相同的数据路径与实验控制参数**。没有本地权重的 Add 实验再增加 `"DualHGNetv2_Add.pretrained=False"`；其他分支使用各自的注册类名。

### 3.4 Linux 单机分布式示例

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7001 --nproc_per_node=2 train.py \
  -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml \
  --use-amp --seed 0 --output-dir outputs/atrumod_add_2gpu_seed0
```

这个示例要求你已经准备好适合本机的路径配置，或追加前述 `-u` 数据路径覆盖项。`total_batch_size` 是全局 batch，必须能被进程数整除。多机代码的 GPU 索引使用存在风险，不把此单机例子当作已验证的多机方案；Windows 优先使用单进程训练。

### 3.5 恢复训练、迁移权重与只验证

下面以 Add 实验为例，均需使用与你的数据路径一致的配置或追加路径覆盖：

```powershell
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml -r outputs/atrumod_add_seed0/last.pth --use-amp -u DualHGNetv2_Add.pretrained=False
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml -t outputs/atrumod_add_seed0/best_stg1.pth --use-amp -u DualHGNetv2_Add.pretrained=False
python train.py -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml -r outputs/atrumod_add_seed0/best_stg1.pth --test-only --use-amp -u DualHGNetv2_Add.pretrained=False
```

- `-r/--resume`：恢复模型及可用的优化器、调度器、EMA、scaler、epoch 等完整训练状态。
- `-t/--tuning`：迁移可匹配的模型权重，不等价于继续原训练状态；形状不匹配的层需重新初始化。
- `--test-only`：调用 `DetSolver.val()`；其名称不保证验证集就是独立的官方 test。
- `--resume` 与 `--tuning` 不能同时使用。
- `train.py` 自动关闭重复预训练的逻辑只覆盖 `HGNetv2`、`DualHGNetv2_InfraNet`。其他骨干仍需显式覆盖 `pretrained=False`。

输出通常包括 `last.pth`、阶段最佳 `best_stg1.pth`/`best_stg2.pth`、周期 checkpoint 和 `log.txt`；具体是否生成取决于阶段、配置和评估结果。当前 ATR 36 轮配置不会进入 `stop_epoch=36` 之后的第二阶段。

### 3.6 容易误用的命令行行为

| 行为 | 实际规则 | 建议 |
| --- | --- | --- |
| YAML 覆盖 | include 先合并，当前 YAML 覆盖，再应用 `-u` 和非 None 的 argparse 值 | 阅读最终打印的配置，不能只看叶子 YAML |
| AMP | `--use-amp` 默认 False 会覆盖 YAML 中的 True | 需要 AMP 时显式加 `--use-amp`，否则各实验可能不是同一数值设置 |
| 学习率调度 | 实际开关拼写是 `lrsheduler` | 不要自行写成 `lr_scheduler` 并误以为已启用 FlatCosine |
| GPU 索引 | 部分 AMP 代码把 `str(device)` 直接作为 `device_type` | 优先用 `CUDA_VISIBLE_DEVICES` 选择 GPU，避免未经验证的 `--device cuda:0` + AMP 组合 |
| CPU 调试 | 可以指定 `--device cpu` | 小规模调试不要开启 CUDA AMP；完整模型 CPU 训练可能很慢 |
| 推理工具 | 没有独立的目录批量推理/ONNX 导出 CLI | `deploy()` 和后处理的导出接口不等于已经有完整部署脚本 |

<a id="datasets"></a>
## 4. 数据集与预处理

### 4.1 ATR-UMOD 成对数据

对应实现为 `engine/data/dataset/atrumod_dataset.py` 的 `ATRUMODPairDataset`。类别顺序是：

```text
car, suv, van, bus, freight_car, truck, motorcycle,
trailer, excavator, crane, tank_truck
```

示意目录如下；`val` 可以是你从训练数据中规范划出的验证集：

```text
ATR-UMOD/
  train/
    images_ir/
    images_rgb/
    labels_ir/
  val/
    images_ir/
    images_rgb/
    labels_ir/
  test/
    images_ir/
    images_rgb/
    labels_ir/
```

IR 图像与 RGB 图像需要对应同一帧。代码从 IR 目录旁查找 `images`、`images_rgb`、`images_color`、`rgb` 等候选目录，并尝试去掉目录名中的 `_ir`；也支持用 `img_folder_rgb` 明确指定。**自动找到目录只证明文件路径匹配，不证明图像已完成几何配准。**

XML 对目标支持两类标注：

- `robndbox` 中的 `cx/cy/w/h/angle`：当前代码将角度按**度**解释；数据源若实际给弧度，会产生系统性错误。
- 四点多边形 `x1/y1/.../x4/y4`：转为 OpenCV 最小外接旋转矩形，再规范长边和角度。

条件信息来自 XML，包含高度、飞行角、时间、天气、光照和位置等。缺失时，高度默认 120、飞行角默认 45、时间默认中午，其归一化时间表示为约 `(0,-1)`；其他条件使用代码中的回退类别。默认值与真实观测目前没有有效性标记。

检查实际 XML 分布可运行：

```powershell
python data_pre/inspect_atrumod.py D:/datasets/ATR-UMOD/train/labels_ir
```

这个脚本用于观察标签字段和频次，不会自动证明类别映射、IR/RGB 配准或所有文件都合格。

### 4.2 DroneVehicle 成对数据

对应 `engine/data/dataset/dota_dataset.py` 中的 `DroneVehiclePairDataset`。类别顺序为：

```text
car, truck, bus, van, freight_car
```

典型原始目录与转换后目录：

```text
DroneVehicle/
  train/
    trainimg/
    trainimgr/
    trainlabelr/
    dota_labels_ir/
  val/
    valimg/
    valimgr/
    vallabelr/
    dota_labels_ir/
  test/
    testimg/
    testimgr/
    testlabelr/
    dota_labels_ir/
```

其中 `*imgr` 是 IR 目录，加载器通过去掉末尾的 `r` 推断 RGB 目录，例如 `trainimgr → trainimg`。Windows 推荐使用正斜杠路径，避免尾部反斜杠影响字符串路径规则。

原始 IR XML 可以转为 DOTA 文本：

```powershell
python data_pre/dronevehicle2dota.py --data-root D:/datasets/DroneVehicle --splits train,val --output-dirname dota_labels_ir
```

转换脚本处理类别名规范化、支持的框格式、边界裁剪和退化框过滤，并输出每类数量及跳过原因。它会写入每个 split 的输出目录，同名 `.txt` 会被覆盖；如果要保留既有标签，先使用新的输出目录名。

每个 DOTA 标签行格式为：

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

前八个数是像素坐标，目标角度由四点转换得到。当前 DOTA 数据加载路径没有实现 difficulty 的完整忽略评估语义，不能假设难例已按官方协议排除。

### 4.3 当前默认划分存在实验协议风险

| 数据配置 | 当前训练目录 | 当前验证目录 | 问题 |
| --- | --- | --- | --- |
| `configs/dataset/atrumod_rgbt.yml` | ATR `train` | ATR **`test`** | 如果按此验证集挑最佳 epoch，就已经在 test 上做模型选择 |
| `configs/dataset/dronevehicle_rgbt.yml` | DroneVehicle **`test/testimgr`** | DroneVehicle `val/valimgr` | 与通常的 train/val/test 命名相反，必须确认这是有意重划分还是配置遗留 |

这些文件还包含 `/home/reverse/workspace/data/...` 的机器相关绝对路径。运行前必须明确自己的数据协议，再覆盖路径；**本次未擅自改动这些配置，也不把原记录直接当作规范独立测试结果。**

### 4.4 成对变换的关键作用

- **几何同步**：两模态共用缩放、翻转、裁剪和 Mosaic 的抽样参数，使同一 target 能对应两幅图。
- **RGB-only 光度增强**：颜色扰动只施加于 RGB，不给热红外人为引入可见光颜色变化。
- **框更新**：旋转框转多边形进行几何变换，再用 `poly2obb` 重建五维框；非等比缩放后，重拟合旋转矩形只是对变形四边形的矩形近似。
- **清理**：移除过小或退化目标，必须同时筛选 labels 等对应字段，防止标签错位。
- **归一化**：最后再转六通道 Tensor，并按 W/H 分别归一化框。

把 RGB 简单 resize 成与 IR 一样大，不等于校正两传感器视差、畸变、时间差或局部错位。若想研究跨模态对应关系，需要额外检验配准质量。

### 4.5 训练前的数据审计清单

1. 比较 split 之间的帧 ID、视频段和场景，避免同一序列或近重复图像跨 train/val/test。
2. 检查每个 IR 是否有对应 RGB、XML/TXT，能否解码，以及两模态是否同帧。
3. 核对类别顺序、未知类别、空标注、难例处理和条件标签缺失率。
4. 抽样可视化 OBB，重点检查角度单位、长边交换、越界框、resize 后框的位置。
5. 检查条件分布：默认值、截断值、地点 ID 与类别是否高度相关。
6. 当前加载器读样本失败时会随机重试其他样本，验证阶段也可能如此；正式评估前应离线发现并处理坏样本，核对最终唯一评估 ID 数量。

<a id="configs"></a>
## 5. 全部配置文件说明

### 5.1 配置合并与对象构建规则

`__include__` 的路径相对于**声明它的 YAML 所在目录**解析。多个 include 按列表顺序合并，之后当前文件覆盖已有字段。

- 字典递归合并；没有显式覆盖的子键会从基础配置保留下来。
- 列表整体替换，不是按索引逐项合并；覆盖 `ops` 就替换整条变换流水线。
- `-u key.subkey=value` 通过 YAML 值解析得到布尔、数值、列表等，再合并入配置。
- 命令行中非 None 的普通 argparse 值还会覆盖配置，包括默认 False 的某些布尔开关。
- `__share__` 表示从顶层共享参数，如 `num_classes`；`__inject__` 表示需要继续构建的子组件，如 backbone、encoder、matcher。

因此，叶子 YAML 中只写 `policy: {name: default}`，并不会删除基础 policy 的 `epoch` 和 `ops`。审查后可以确认：**当前主要配置继承了这些键，不会因为缺少 `epoch` 而立即触发 Compose 的日志异常**；独立手写一个仅含 name 的 policy 则不一定安全。

### 5.2 运行、数据和基础配置：6 个文件

| 文件 | 作用与当前使用方式 |
| --- | --- |
| `configs/runtime.yml` | 规定日志频率、输出目录、checkpoint 间隔、SyncBN、AMP scaler、EMA 等运行默认项。基础值会被实验 YAML 覆盖；注释中的 SwanLab 配置不代表已启用记录。 |
| `configs/base/dataloader.yml` | 提供通用 DataLoader、增强 Compose、collate 的基础模板，包含 Mosaic/增强阶段 policy。当前 ATR 的成对 ops 替换了普通增强列表，但未覆盖的 policy 子键仍会继承。 |
| `configs/base/rtv2_optimizer.yml` | 配置 AdamW、按参数名正则分组、不同层的学习率/权重衰减、梯度裁剪、EMA 和调度默认项。保留 MultiStepLR 与 warmup 配置；实验的 `lrsheduler` 非空时实际转用迭代级 FlatCosine。 |
| `configs/base/rtv2_r50vd.yml` | 原始水平检测模型模板：RTv4、PResNet、HybridEncoder、RTDETRv2 解码器、水平框后处理及标准 criterion/matcher。OBB 实验覆盖了实际组件名和通道参数，不能把模板中的 ResNet 当成当前在运行的骨干。 |
| `configs/dataset/atrumod_rgbt.yml` | 11 类 ATR 成对数据集、训练/验证加载器和本地 `DotaEvaluator`。含硬编码路径及 train/test 划分，需要本地覆盖和协议审查。 |
| `configs/dataset/dronevehicle_rgbt.yml` | 5 类 DroneVehicle 成对数据集和本地旋转框评估器。默认训练指向 test，验证指向 val；该选择不能不加说明地用于论文结论。 |

### 5.3 ATR 实验配置：6 个文件

以下文件均在 `configs/rtdetrv2_obb/`，均继承 ATR 数据、runtime、dataloader、optimizer 和基础模型模板。

| 文件 | 选用组件 | 实验目的与特殊设置 |
| --- | --- | --- |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_single_ir.yml` | `RTv4 + SingleHGNetv2_IR` | 单 IR 基线；数据仍六通道输入，骨干只取前三通道。 |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml` | `RTv4 + DualHGNetv2_Add` | 三尺度直接相加的双流基线。 |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_concat_baseline.yml` | `RTv4 + DualHGNetv2_Concat` | 三尺度拼接、1×1 投影的双流基线。 |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_prototype.yml` | `RTv4 + DualHGNetv2_LGDense` | `cond_dim=256`，启用六组条件预测头及 `loss_cond`，外部权重 0.5。 |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_self_driven.yml` | 同上 | `with_aux_heads=False`，去掉条件标签监督；联合条件编码器、RGB 局部门和全局通道门仍保留。 |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_infranet.yml` | `RTv4_Infranet + DualHGNetv2_InfraNet + RTv4OBBCriterion_Infranet` | 主/辅助检测训练，质量门融合；**实际 `eval_mode: rgb_ir`**，不是类默认的 IR-only。 |

共同的主要有效参数如下：

| 参数组 | 当前 ATR 设置 | 含义 |
| --- | --- | --- |
| 图像 | `[512,640]`，H/W 顺序 | 不是正方形，直接影响旋转几何审查 |
| 训练 | `epoches=36`；训练全局 batch 8、验证 16 | 从零开始 epoch 通常为 0～35 |
| 编码器/解码器 | `hidden_dim=224`、8 个注意力头、3 层解码 | 相比基础模板缩小模型，不能只按文件名推断规模 |
| 查询 | 普通查询 300，去噪预算 100 | 实际 DN 数随 batch 最大目标数和分组变化，不一定恰好为 100 |
| 检测损失 | focal/L1/KLD 权重 1/5/5 | L1 包含第五个角度分量；角度目前非周期处理 |
| 匹配 | class/bbox/Hausdorff/`giou` 权重 2/0/5/5 | `giou` 名称在 OBB matcher 中实际对应 KLD，不是旋转 GIoU |
| 回归匹配集合 | `use_uni_set=True` | 聚合多层匹配对用于回归；分类仍使用各层自身匹配 |
| 调度 | warmup 500 iter、flat 18 epoch、no-aug 0、gamma 0.1 | 由 `lrsheduler: flatcosine` 触发 |
| 增强 | Resize→Sanitize→Tensor→ConvertOBB | **当前 ATR 没有随机几何/光度增强** |
| collate | base_size 640、repeat 为 null、mixup 0、stop_epoch 36 | 多尺度与 mixup 实际关闭；base_size 不会自动把输入变成 640×640 |
| AMP/EMA | YAML 为开启 | AMP 仍受命令行默认 False 的覆盖行为影响 |

512×640 的像素面积比 640×640 少 **20%**；反过来 640×640 比 512×640 多 25%。两种说法基准不同，不能把面积差直接当作完全相同百分比的 FLOPs/FPS 收益。

### 5.4 DroneVehicle 实验配置：2 个文件

| 文件 | 作用与有效设置 |
| --- | --- |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_dronevehicle_concat.yml` | 完整双流 concat 实验。640×640、142 轮、训练/验证全局 batch 均为 4，3 层解码、224 hidden。warmup 2000 iter、flat 70 epoch、no-aug 12。训练包含成对 Mosaic、RGB-only 光度扰动、zoom/crop/flip；policy 阶段 `[4,82,120]`、Mosaic 概率 0.5。多尺度和 mixup 关闭。训练普通 resize，验证等比缩放并在右/下补边。 |
| `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_dronevehicle_tsti_sup_fast.yml` | 遗留/未完成实验声明。640×640、36 轮、请求 `AsymHGNetv2_TSTI_Sup` 和 saliency 损失；当前源码没有该骨干，OBB criterion 也没有该 loss 映射。不能据此宣称 TSTI/显著性监督已实现或取得效果。 |

Drone 的 Compose 强增强停止 epoch 120 与 scheduler 的 no-aug 12 是两个不同控制项，不应混为同一个时间点。`rotation_range=0` 也意味着当前 Mosaic 不会检验非零旋转方向是否正确。

<a id="file-atlas"></a>
## 6. 全部 Python 文件逐一说明

下表逐一列出全部 97 个 Python 文件。“主链路”表示当前主要 OBB 实验会用到；“实验分支”表示需要选中对应 YAML；“保留”表示源码存在但当前配置未使用，不能因此假设已经完整验证。

### 6.1 根目录、包入口与数据准备：4 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `train.py` | 总入口：解析参数、初始化分布式与随机种子、构建 YAMLConfig、按 task 选择 solver，执行训练或验证并清理进程组。还保留可选远程 DOTA 任务管理。 | 主链路。不是模型结构文件；AMP 和预训练覆盖的实际行为应从这里开始理解。 |
| `engine/__init__.py` | 导入 core、optim、data、rtv4、backbone 等子包，触发装饰器注册，使 YAML 中的类名可被查找。 | 初始化基础设施。导入时也会加载部分未使用模块的依赖。 |
| `data_pre/dronevehicle2dota.py` | 把 DroneVehicle IR XML 标注转成 DOTA 四点文本，规范类别、裁剪检查、过滤退化框、报告转换统计。 | 数据准备 CLI；会写输出标签，不能对已有标签目录无备份覆盖。 |
| `data_pre/inspect_atrumod.py` | 扫描 ATR XML，统计条件字段和取值，帮助确定数据标注的真实结构和分布。 | 只读检查脚本；不是训练数据集，也不是自动质量验收程序。 |

其他根目录文件：`requirements.txt` 声明依赖但存在语法问题；`experiment.md` 保存已有实验笔记，不是自动生成的完整训练日志；`.gitignore` 是本地忽略规则；`LICENSE` 是根许可证；`pretrain/hgnetv2/PPHGNetV2_B0_stage1.pth` 是二进制预训练权重，不是代码。缓存 `.pyc` 不是源文件，不应编辑，也不应据它判断源码实现。

### 6.2 骨干与融合模块：17 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/backbone/__init__.py` | 汇总导入骨干、封装器和融合组件，完成 YAML 注册入口。 | 新增骨干只有类定义还不够，通常也需让该包导入它；本次没有新增模型。 |
| `engine/backbone/common.py` | 提供卷积+归一化、激活选择、FrozenBatchNorm2d 与冻结 BN 的递归替换等公共部件。 | 多个骨干共用，改变它可能影响多条实验路径。 |
| `engine/backbone/utils.py` | `IntermediateLayerGetter` 按指定中间层名称提取特征，适配外部网络的多尺度输出。 | 基础适配工具，不直接执行多模态融合。 |
| `engine/backbone/hgnetv2.py` | 实现 HGNetv2 的 stem、轻量卷积、密集 HG block、ESE、stage、预训练加载及冻结设置。 | 当前所有主要融合骨干的基础网络；`use_lab` 是 Learnable Affine Block，不是 Lab 色彩空间。 |
| `engine/backbone/single_hgnetv2_ir.py` | 用一套 HGNetv2 提取 IR，统一三尺度输出接口；六通道时截取前三通道。 | 单 IR 基线。成对 loader 的 RGB 文件需求并没有因此消失。 |
| `engine/backbone/dual_hgnetv2_add.py` | 两套 HGNetv2 分别提取 IR/RGB，按 S3/S4/S5 对应位置相加。 | 实验分支。要求对应尺度、通道和空间尺寸相同，输入已配准。 |
| `engine/backbone/dual_hgnetv2_concat.py` | 双流特征拼接，用每尺度 1×1 Conv/BN/SiLU 压回检测编码器期望的通道数。 | 实验分支。比 Add 多了融合层，做公平比较要记录参数/计算量。 |
| `engine/backbone/dual_hgnetv2_lg_dense.py` | `ConditionPerceiver` 联合编码条件，`LocalGlobalGatedDenseFusionBlock` 执行全局通道与 RGB 局部空间门控，输出融合特征及可选条件预测。 | prototype/self-driven。这里的“条件”与“质量”是学习到的代理，不是经过传感器标定的物理量。 |
| `engine/backbone/qual_gate.py` | `ChannelAdapter` 对齐不同尺度 IR 特征的通道/分辨率；`QualGate` 计算图像级 q、IR 增益 alpha、多尺度通道门和输出融合。 | InfraNet 的核心门控单元；没有直接的质量真值监督。 |
| `engine/backbone/dual_hgnetv2_infranet.py` | 完整 IR HGNetv2 + 小型 RGB stem + 三个 QualGate，组织主 IR 和辅助融合特征，并支持 `ir`/`rgb_ir` 推理选择。 | InfraNet 分支；训练返回字典，与普通骨干的特征列表接口不同。 |
| `engine/backbone/presnet.py` | Paddle 风格 ResNet/ResNet-vd 实现，提供多尺度输出、冻结和预训练入口。 | 保留的通用骨干；基础 YAML 使用它，但当前 OBB 叶子配置已替换。 |
| `engine/backbone/csp_resnet.py` | CSPResNet、有效 SE、RepVGG 风格可重参数化 block，提供替代特征提取网络。 | 保留；改变骨干时要同步 encoder 的通道与步长。 |
| `engine/backbone/csp_darknet.py` | CSPDarkNet 提取特征，CSPPAN 进行路径聚合；包括 C3、SPPF 等构件。 | 保留的骨干/颈部方案，不是当前 HGNetv2 主路径。 |
| `engine/backbone/lsknet.py` | 大选择核注意力网络 LSKNet，含重叠 patch embedding、深度卷积及局部/大核选择。 | 保留的遥感骨干候选，不代表已在本项目 RGBT 实验中评估。 |
| `engine/backbone/timm_model.py` | 封装 timm 的特征网络，按配置创建外部骨干并返回所需中间特征。 | 保留；实际可用模型名、通道和权重由 timm 版本决定。 |
| `engine/backbone/torchvision_model.py` | 使用 torchvision 模型与特征提取接口，适配多尺度骨干输出。 | 保留；不能默认支持六通道 RGB-IR 输入或任意层名。 |
| `engine/backbone/test_resnet.py` | 定义 `_ResNet`、残差块和普通 ResNet 分类封装 `MResNet`。 | **名字带 test，但它是模型实现文件，不是自动化测试套件**；不是当前 OBB 检测骨干，也不因 M 前缀自动成为多模态网络。 |

### 6.3 配置、注册与依赖注入：5 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/core/__init__.py` | 导出注册表、配置和构建函数，为其他包提供统一导入接口。 | 初始化基础设施，不包含训练循环。 |
| `engine/core/_config.py` | `BaseConfig` 定义设备、模型、训练轮数、加载器、优化器等公共配置属性和默认行为。 | YAMLConfig 的基础抽象；不能只看 YAML 而忽略对象属性默认值。 |
| `engine/core/workspace.py` | `register` 记录组件及构造签名；`extract_schema` 提取默认参数；`create` 执行共享参数、依赖注入与实例化。 | 主链路。组件名缺失、参数名错误会在这里暴露，不会自动猜测替代实现。 |
| `engine/core/yaml_utils.py` | 读取相对 include、递归字典合并、解析命令行点路径和值、展开构建配置。 | 主链路；字典深合并、列表替换是理解实验配置的关键。 |
| `engine/core/yaml_config.py` | 把 YAML 转成惰性构建的模型、criterion、postprocessor、optimizer、loader、evaluator、EMA/scaler 等对象。还按参数正则分组和全局 batch 计算本进程 batch。 | 主链路。属性第一次访问可能才触发加载权重或创建数据集。 |

### 6.4 数据公共层：3 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/data/__init__.py` | 汇总数据集、变换和 DataLoader 导入，使数据相关类可由 YAML 构建。 | 初始化入口，也会带入保留数据集的依赖。 |
| `engine/data/_misc.py` | 包装 torchvision `tv_tensors`，提供图像、框、掩码到带元数据 Tensor 的转换辅助。 | 普通/旋转变换的公共工具；须区分五维 OBB 与标准四维 BoundingBoxes。 |
| `engine/data/dataloader.py` | 自定义 DataLoader 的 epoch 传递；实现简单 collate 和 `BatchImageCollateFunction`，负责堆叠、多尺度、mixup、增强停止信息。 | 主链路。target 保持按图像组成的列表，条件字典随 target 保留；当前主要配置关闭多尺度/mixup。 |

### 6.5 数据集与评估器：16 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/data/dataset/__init__.py` | 导出各数据集与评估器，触发注册。 | 初始化入口；某个评估器存在不代表当前配置选中了它。 |
| `engine/data/dataset/_dataset.py` | `DetDataset` 定义检测数据集基础接口，并支持把 epoch 传给变换。 | 增强调度依赖它与 DataLoader 的协作。 |
| `engine/data/dataset/atrumod_dataset.py` | ATR 成对图像匹配、XML/多边形旋转框解析、11 类映射、六组条件解析和评估真值接口。 | ATR 主链路；角度单位、条件缺省和随机重试需要审计。 |
| `engine/data/dataset/dota_dataset.py` | 定义通用 DOTADataset、DroneVehicle 单模态与成对加载器；解析 DOTA 四点文本、拟合 OBB、组织 target 和类别。 | DroneVehicle 主链路及保留 DOTA 路径。末尾 r 推断 RGB、difficulty 未完整处理。 |
| `engine/data/dataset/dota_evaluator.py` | 本地旋转检测评分：收集每图预测、跨进程汇总、获取 dataset 真值，调用 mmeval 计算 AP，提供 solver 所需 stats。 | 当前 ATR/Drone 的真实本地评估；默认 AP50，不是默认 COCO AP50:95。 |
| `engine/data/dataset/mmeval.py` | 自包含旋转框/四边形几何与评价：多边形交集、IoU、NMS、按分数匹配 TP/FP、AP 积分，以及可选 Numba/多进程加速。 | OBB 本地评估底层；这里的 IoU 与训练 KLD 是两种不同量。 |
| `engine/data/dataset/dota_auto_evaluator.py` | 在线/离线 DOTA 结果导出、切片坐标恢复与合并、类别文本打包 ZIP，以及可选远程提交。 | 非当前本地评估路径；其返回 `mAP` 使用 epoch 占位，不能当真实 AP。会重建结果目录，使用前必须确认路径安全。 |
| `engine/data/dataset/hrsc_dataset.py` | 读取 HRSC2016 舰船数据、组织旋转框标注和检测 target。 | 保留的单类遥感数据接口；没有当前 RGBT 实验配置。 |
| `engine/data/dataset/hrsc_evaluator.py` | 组织 HRSC 数据的旋转检测预测与真值评价。 | 保留的专用评估适配器，需与 HRSC loader/后处理配套验证。 |
| `engine/data/dataset/dior_dataset.py` | 读取 DIOR 遥感检测数据，提供 20 类映射、框与样本元数据。 | 保留；格式与当前 ATR 条件监督数据不同。 |
| `engine/data/dataset/fair1m_dataset.py` | 读取 FAIR1M 细粒度遥感目标数据，处理 37 类与旋转标注。 | 保留；不能直接沿用 ATR 的类别数和标签顺序。 |
| `engine/data/dataset/coco_dataset.py` | 基于 torchvision COCO 读取图像/标注，转水平框、可选分割掩码、COCO 类别映射及 target。 | 保留 HBB/分割路径；其 `orig_size` 约定与 OBB 路径不同。 |
| `engine/data/dataset/coco_eval.py` | 封装 faster-coco-eval 的 COCO bbox/segm 等评分、预测格式转换及跨进程合并。 | 保留的 COCO 评估，不是当前 DotaEvaluator 的 AP 计算实现。 |
| `engine/data/dataset/coco_utils.py` | 将通用检测数据转成 COCO API，处理 polygon mask、空/无效标注和数据集包装。 | HBB/COCO 辅助工具，不能不经改造接收五维 OBB。 |
| `engine/data/dataset/voc_detection.py` | VOC XML/图像读取，输出常规检测 target。 | 保留的数据接口，不包含 ATR 六组环境条件。 |
| `engine/data/dataset/voc_eval.py` | 提供 VOCEvaluator 的基础接口占位。 | 当前不是完整实现的 VOC AP 评估器，不能据此承诺完整 VOC 训练评测。 |

### 6.6 图像与框的变换：8 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/data/transforms/__init__.py` | 导入并注册普通、OBB、成对变换和 Compose。 | 数据流水线的初始化入口。 |
| `engine/data/transforms/_transforms.py` | 注册 torchvision 常用变换，补充空变换、PadToSize、IoUCrop、框格式转换和 PIL→Tensor 等通用算子。 | 主要为普通单图/HBB 路径；成对数据不能随意换回普通几何变换。 |
| `engine/data/transforms/container.py` | `Compose` 按 YAML 创建 ops，根据 epoch policy 调度增强，并处理 Mosaic 与 crop/zoom 等组合关系。 | 主链路。增强是否执行由 policy 和概率共同决定，不是 ops 中写了就每次执行。 |
| `engine/data/transforms/functional.py` | 提供 crop、hflip、resize、pad、interpolate 等图像与常规框几何函数。 | 公共低层工具，涉及框坐标与 canvas 元数据更新。 |
| `engine/data/transforms/mosaic.py` | 普通 Mosaic 多图拼接、几何变换与标注汇总。 | 保留单模态/HBB 增强，不是当前 Drone 的成对 Mosaic。 |
| `engine/data/transforms/mosaic_obb.py` | 单模态旋转框 Mosaic，拼接时同步变换 OBB 多边形并重新拟合。 | 保留 OBB 增强；非零 affine 旋转方向要专门验证。 |
| `engine/data/transforms/transforms_obb.py` | OBB↔polygon、旋转框的 pad/crop/zoom/flip/affine/resize、清理与归一化转换。 | 成对变换复用的基础几何；框与图像的角度方向、非等比缩放需要检查。 |
| `engine/data/transforms/transforms_rgbt.py` | 实现 PairedResize/Flip/Mosaic/ZoomOut/IoUCrop 等锁步几何、RGB-only 光度增强、成对清理、六通道 Tensor 与 OBB 转换。 | ATR/Drone 主链路；输入输出顺序固定为 IR 后 RGB。 |

### 6.7 分布式、日志、几何与可视化：8 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/misc/__init__.py` | 汇总导出分布式、日志和其他公共工具。 | 初始化入口。 |
| `engine/misc/dist_utils.py` | 初始化/清理进程组、随机种子、DDP/SyncBN、分布式 sampler、rank/world size、指标归约和跨进程数据收集。 | 主链路的并行基础设施；多机 global rank 与本机 GPU 索引的关系有风险。 |
| `engine/misc/logger.py` | `SmoothedValue` 统计滑动值与全局均值，`MetricLogger` 打印数据/迭代耗时、loss、学习率等，并同步统计。 | 主链路；日志平滑值不等于完整测试集指标。 |
| `engine/misc/profiler_utils.py` | 按配置输入尺寸深拷贝模型、切部署态、用 calflops 估算 FLOPs/MACs/参数。 | solver 的模型统计。使用三通道 dummy，双流骨干会兼容性复制，不是实测 RGBT 吞吐。 |
| `engine/misc/box_ops.py` | 水平框 GIoU、逐元素 IoU、点与框关系及距离表示。 | 公共 HBB 几何工具，**不是真正的旋转框 IoU 实现**。 |
| `engine/misc/lazy_loader.py` | 延迟导入模块的包装器，减少某些可选依赖的提前导入。 | 辅助基础设施，不保证整个 engine 的所有导入都惰性执行。 |
| `engine/misc/visualizer.py` | 绘制普通检测样本和框，辅助检查数据/增强。 | 调试工具；六通道输入需要先选择要显示的模态。 |
| `engine/misc/vis_obb.py` | 将归一化角度转换为弧度、构造旋转矩形并可视化训练 OBB。 | 检查标注与几何增强；应明确使用当前尺寸而不是错误的原图尺寸。 |

### 6.8 优化器、AMP、EMA 与学习率：6 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/optim/__init__.py` | 导入优化器、scheduler、warmup、AMP 和 EMA 组件。 | 初始化/注册入口。 |
| `engine/optim/optim.py` | 将 PyTorch 优化器、常见学习率调度器注册到 YAML 构建系统。 | 主链路使用 AdamW；不是自行重写 AdamW 更新公式。 |
| `engine/optim/amp.py` | 注册 GradScaler，让配置可以创建自动混合精度的缩放器。 | AMP 基础部件；实际 autocast、scale/backward/step 在训练循环中执行。 |
| `engine/optim/ema.py` | `ModelEMA` 维护模型参数/缓冲区的指数滑动平均，支持 warmup、state 保存恢复；另保留 ExponentialMovingAverage 实现。 | 当前实验启用 ModelEMA。修改对象的 decay 属性不一定改变闭包捕获的真实更新率。 |
| `engine/optim/lr_scheduler.py` | 实现带线性预热、平坦阶段、余弦下降及末段最低率的迭代级 FlatCosine 调度。 | 由 `lrsheduler` 开关选用；轮数参数转换成迭代数。 |
| `engine/optim/warmup.py` | Warmup/LinearWarmup 在正式 scheduler 前渐进提高学习率。 | 通用调度辅助；不要与 FlatCosine 内置 warmup 重复计算。 |

### 6.9 模型、解码器、匹配与损失：21 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/rtv4/__init__.py` | 导入/注册 RTv4 容器、编码器、解码器、criterion、matcher、postprocessor 和 teacher。 | 初始化入口；保留组件可能引入额外导入依赖。 |
| `engine/rtv4/rtv4.py` | 组合 backbone→encoder→decoder；兼容普通特征列表与含条件预测的骨干字典，训练时把条件输出传给 loss。提供部署转换。 | Add/Concat/SingleIR/LGDense 主容器，不在这里定义融合公式。 |
| `engine/rtv4/rtv4_infranet.py` | 处理 InfraNet 主/辅助特征字典，训练让同一 encoder/decoder 分别处理两个流，验证只处理选定输出；透传质量诊断。 | InfraNet 专用容器。共享权重不等于只执行一次前向。 |
| `engine/rtv4/hybrid_encoder.py` | 通道投影、AIFI Transformer、top-down FPN 与 bottom-up PAN，含卷积/RepVGG 风格融合部件和可选蒸馏投影。 | 主链路三尺度编码器；二维位置编码与 flatten 索引关系存在已验证问题。 |
| `engine/rtv4/rtdetrv2_obb_decoder.py` | 五维 OBB 查询初始化、编码器 top-k 候选、旋转可变形注意力、多层框 refinement、去噪与辅助预测。 | 核心 OBB 检测头。非正方形输入下的旋转采样度量问题见第 8 节。 |
| `engine/rtv4/denoising_obb.py` | 按 GT 构造正/负去噪查询，加入类别及 xywh 噪声，填充变长目标并建立注意力屏蔽矩阵。 | 训练专用；当前不扰动第五维角度，去噪预算不等于固定查询数。 |
| `engine/rtv4/obb_matcher.py` | 计算分类、五维 L1、Gaussian KLD、四角点集合 Hausdorff 成本，再用 SciPy Hungarian 求每张图的指派。 | 训练正负样本分配；不是验证时按 score/IoU 判定 TP/FP 的算法。 |
| `engine/rtv4/rtv4_obb_criterion.py` | focal 分类、L1/KLD 回归、辅助层/编码器/DN 损失、uni-set 回归匹配与可选六组条件损失。 | 普通 OBB 训练损失；角度 L1 非周期、NaN 归零和 aux_outputs 前提需注意。 |
| `engine/rtv4/rtv4_obb_criterion_infranet.py` | 独立实现同类 OBB 损失，并分别计算主/辅助检测损失，辅助损失固定乘 0.25，再汇总。 | InfraNet 专用 criterion，不是直接继承普通 criterion 的薄子类；没有显式 teacher-student 蒸馏项或质量标签校准项。 |
| `engine/rtv4/postprocessor.py` | 同文件包含 HBB PostProcessor 与 OBB PostProcessorOBB，负责分数 top-k、像素还原、角度转弧度及导出输出。 | 当前使用 OBB 类；不默认做 NMS，不适配任意 padding/缩放策略。 |
| `engine/rtv4/utils.py` | 正余弦坐标编码、inverse_sigmoid、激活、可变形注意力采样/聚合核心等公共函数。 | 核心数值工具；grid_sample 的归一化采样坐标不等于像素度量。 |
| `engine/rtv4/utils_obb.py` | OBB 角度转换、框→二维 Gaussian 均值/协方差、KLD、四角点生成与 Hausdorff 距离。 | matcher/criterion 的几何基础；归一化矩形图像的协方差度量一致性需要审查。 |
| `engine/rtv4/rtdetrv2_decoder.py` | 原始四维水平框 RT-DETRv2 解码器、可变形注意力和查询选择。 | 保留 HBB 检测头；不是当前五维 OBB 实现。 |
| `engine/rtv4/denoising.py` | 水平框版本的类别/框去噪查询与注意力掩码。 | 保留 HBB 训练工具，不含旋转角。 |
| `engine/rtv4/matcher.py` | 水平检测的分类/L1/GIoU Hungarian 匹配器。 | 保留，不能把其 GIoU 与 OBB matcher 中名为 `cost_giou` 的 KLD 混同。 |
| `engine/rtv4/rtv4_criterion.py` | 水平检测标准 criterion，包含多种分类/框损失、辅助匹配及基础模型蒸馏接口。 | 保留 HBB 路径。其 distill loss 映射不代表 OBB criterion 已支持同样配置。 |
| `engine/rtv4/box_ops.py` | cxcywh/xyxy 转换、水平 IoU/GIoU、mask→box。 | HBB 工具，旋转评估应使用 mmeval 等真正 OBB 几何。 |
| `engine/rtv4/dfine_decoder.py` | D-FINE 风格四维框解码器：离散分布回归、积分、定位质量估计、逐层 refinement。 | 保留实验组件，没有当前 OBB/六通道主配置；并未自动扩展角度分布。 |
| `engine/rtv4/dfine_utils.py` | D-FINE 的距离分布权重、GT bin 编码、距离↔水平框转换。 | 为 D-FINE 服务，不是旋转框的长边/角度工具。 |
| `engine/rtv4/dinov2_teacher.py` | 加载/冻结 DINOv2 teacher，提取 patch 特征供可选基础模型蒸馏，支持按高宽还原 patch 网格。 | 当前 OBB 配置未启用。teacher 预期三通道输入，不能直接接收整个六通道 batch。 |
| `engine/rtv4/dinov3_teacher.py` | DINOv3 teacher 特征提取与蒸馏适配。 | 当前未启用；实现中的 token 网格还原有正方形假设，矩形输入需另行验证。 |

### 6.10 训练与验证控制器：6 个文件

| 文件 | 在项目中的职责 | 使用场景/注意事项 |
| --- | --- | --- |
| `engine/solver/__init__.py` | 导出 solver，并用 TASKS 把 `detection` 等任务名映射到对应控制器。 | `train.py` 选择 DetSolver 的入口。 |
| `engine/solver/_solver.py` | `BaseSolver` 负责模型/criterion 到设备、DDP/EMA/AMP/优化器/loader 初始化、完整 checkpoint 保存恢复和 tuning 权重加载。 | 主链路的状态管理，resume 与 tuning 在这里真正分开。 |
| `engine/solver/det_solver.py` | 检测 fit/val：组织 epoch、scheduler、增强阶段切换、训练→验证、best/last/周期保存与日志。 | 主链路；阶段转换可能重新载入最佳状态，必须审查 resume 和 stage-2 行为。 |
| `engine/solver/det_engine.py` | 单 epoch 的训练/验证细节：递归移动 target、前向、loss 汇总、AMP、梯度裁剪、EMA、日志及 postprocessor/evaluator 调用。 | 主链路；字典递归搬运是六组条件进入 GPU 的关键。 |
| `engine/solver/clas_engine.py` | 保留分类任务的单 epoch 训练和准确率验证函数。 | 不是 OBB 主路径；旧接口假设需独立联调，不能承诺直接可运行。 |
| `engine/solver/clas_solver.py` | 保留分类训练控制器与状态/循环组织。 | 当前没有配套分类实验 YAML，不是本项目默认任务。 |

### 6.11 已有检查脚本：3 个文件

| 文件 | 在项目中的职责 | 能验证什么、不能验证什么 |
| --- | --- | --- |
| `tools/test_qual_gate.py` | 构造多尺度模拟特征，检查 QualGate 前向、输出尺寸、q/alpha 范围和反向梯度。 | 能检查模块能否计算；不能证明 q 与真实夜间/噪声质量单调相关。 |
| `tools/test_qual.py` | 构建 InfraNet 骨干，检查训练字典接口、IR-only/RGB-IR 两种 eval 输出与梯度传播。 | 能检查两条路径；不能证明 IR-only 已移除所有 RGB 参数或速度一定提高。 |
| `tools/test_all.py` | 用合成输入/target 构建完整 InfraNet 检测器和 criterion，检查主/辅助检测损失和反向传播。 | 能做端到端 smoke test；不能代替数据集 AP、鲁棒性或论文复现实验。 |

这些脚本头部 docstring 中的 `tests/...` 是旧路径说明；实际文件都位于 `tools/`，仓库没有对应 `tests/` 目录。

<a id="core-details"></a>
## 7. 主要代码的详细解释

本节配合源码中的中文新增注释阅读。注释重点说明输入输出、模块责任、训练/推理差异、单位约定与审查风险，不通过改函数或改参数“顺便修复”问题。

### 7.1 `train.py` 与配置构建：为什么 YAML 能决定网络

`train.py` 本身没有写死某个骨干，它只选择配置和 solver。实际组装主要依赖三个步骤：

1. `register`：在导入模块时记录类/函数名称和构造签名。
2. `YAMLConfig`：加载、合并配置，按需要访问惰性属性。
3. `create`：查注册表，读取共享参数，递归创建注入组件，再实例化外层对象。

以 `RTv4` 为例，其 `backbone`、`encoder`、`decoder` 是注入组件。YAML 先选择 `DualHGNetv2_Add`、`HybridEncoder`、`RTDETRTransformerv2OBB`，再用对应同名配置块提供参数。matcher 同理，由 criterion 的 matcher 配置继续注入。

顶层 `num_classes=11` 会通过共享机制进入需要类别数的组件。修改数据集类别时，不能只改一个预测头而保留错误的 dataset label mapping。

优化器的参数组按名称正则匹配，而不是按“这是我新加的模块”自动分类。新增门控层是否落入低学习率 backbone 分组、BN/norm 是否免衰减，都要检查最终参数名与分组结果。

### 7.2 数据集、变换、collate 与 target 递归搬运

`ATRUMODPairDataset`/`DroneVehiclePairDataset` 负责从磁盘产生一对图像和 target；变换负责同步更新图像与框；collate 负责形成 batch。这三层不能互相代替：

- Dataset 解决“读哪张图、哪份标签、对应哪种类别/条件”。
- Paired transforms 解决“两个模态接受相同几何变化后，目标在哪里”。
- Collate 解决“如何堆叠、是否改变 batch 尺寸、是否 mixup、当前 epoch 是多少”。

target 中重要字段包括 `boxes`、`labels`、`image_id`、`orig_size`、`size`，ATR 还含 `conditions`。每张图目标数可不同，因此 target 是列表，而不是把所有框强行堆成固定 `[B,N,5]`。

`det_engine._to_device` 对 Tensor、字典、列表、元组递归处理，让 `conditions` 中嵌套的数值 Tensor 进入设备；仅写 `{key: value.to(device)}` 的浅层处理无法安全搬运条件字典。

增强调度靠 DataLoader/Dataset/Compose 的 `set_epoch` 协作。不能以为只给 solver 一个 epoch，就自然改变所有 worker 内的数据增强状态。

### 7.3 HGNetv2：单流和双流都依赖的特征底座

`hgnetv2.py` 中的主要层级为：

```text
StemBlock → HG_Stage → HG_Block
                    ├─ 连续卷积/轻量卷积产生中间特征
                    ├─ 拼接多个中间结果形成密集聚合
                    ├─ 压缩与 ESE 通道重标定
                    └─ 按 block 配置使用残差连接
```

当前 B0 的多尺度通道为 `[64,256,512,1024]`。融合包装器使用后三个尺度，使不同融合方案能共享同一 encoder/decoder 接口。

- **Add/Concat/LGDense 是两套独立 HGNetv2 参数**，不是用同一个网络分别处理两模态再共享全部权重。
- 即使两流加载相同预训练初值，后续梯度和 BN 统计也会分开演化。
- `freeze_at`、冻结 BN 等设置会影响哪些参数可学、统计是否更新，不只是显存优化开关。
- `LearnableAffineBlock` 是对特征做可学习仿射变换的小模块；`use_lab` 与图像转 Lab 色彩空间无关。
- 三通道输入被部分双流包装器复制成 IR/RGB 两份，是为兼容模型统计/调试。它不能验证真实双模态互补效果。
- 部分融合包装器硬编码了 B0/B1 的通道或三层输出，不能只修改 `name` 就假设支持任意 HGNetv2 规格。

### 7.4 三种融合：Add、Concat 与 LGDense

设某尺度 IR/RGB 特征为 `I_i, R_i ∈ R^(B×C_i×H_i×W_i)`，位置对齐是基本前提。

**Add 基线：**

```text
F_i = I_i + R_i
```

它不显式估计模态质量，也没有额外融合投影；训练会通过两套骨干自适应调整特征，但无法保证坏 RGB 不干扰 IR。

**Concat 基线：**

```text
F_i = SiLU(BN(Conv1×1(concat(I_i, R_i))))
```

它允许通道混合，比固定相加更灵活，但增加参数/计算和归一化层。与 Add 比较时应同时报告精度和代价。

**LGDense 条件感知融合：**

`ConditionPerceiver` 对两流 S3 特征做全局池化，拼接后经 MLP 得到 `z_cond`。该向量由图像学习得到，不是把 XML 中的真实天气/高度直接塞给推理模型。

每个尺度利用 z 生成 IR/RGB 的全局通道门，同时从 RGB 产生单通道局部空间门。概念公式为：

```text
z = MLP(concat(GAP(I_3), GAP(R_3)))
u_ir_i, u_rgb_i = sigmoid(MLP_i(z))
m_rgb_i = RGB 局部深度卷积门，形状 [B,1,H_i,W_i]
F_i = Conv3×3BNAct(u_ir_i ⊙ I_i + m_rgb_i ⊙ u_rgb_i ⊙ Align1×1(R_i))
```

`⊙` 是带广播的逐元素乘法。全局通道门在一张图内不区分不同位置；局部门区分位置，但其输入主要是 RGB，不构造 IR 位置到 RGB 位置的对应矩阵。

需要准确理解以下结论：

1. **有间接交互**：IR/RGB 共同决定 z，z 再控制两流，不能说完全没有模态互动。
2. **缺少显式局部对应交互**：没有 IR-query/RGB-key 的跨位置注意力，也没有可学习几何对齐。
3. **没有显式恒等 IR 保底**：IR 也被门控，最终又经卷积；不能声称融合一定不会削弱 IR。
4. 六组条件头是图像级多任务监督，不是直接监督“这个位置的 RGB 对检测有帮助”。
5. self-driven 只移除条件预测头/标签损失，z 和门控仍在训练与推理中使用。

### 7.5 InfraNet：质量门与双路共享检测头

`DualHGNetv2_InfraNet` 使用完整 IR HGNetv2 作为主路，RGB 用较浅 stem 得到起始辅助特征，再逐级与 IR 多尺度特征融合。

三处 QualGate 的 IR 输入分别为：

```text
S3 gate：IR S3、S4、S5
S4 gate：IR S4、S5
S5 gate：IR S5
```

`ChannelAdapter` 先做通道/空间适配。QualGate 的关键关系是：

```text
q = sigmoid(MLP(GAP(a_in)))                 每张图一个标量
alpha = clamp(1.5 - q, min=1.0, max=1.5)
a_out = ConvBNAct(q · a_in + Σ_k alpha · g_k ⊙ Adapter(IR_k))
```

其中 `g_k` 为学习到的通道门。q 小时公式降低输入辅助特征的权重，并增加 IR 项的增益；**公式的单调性不等于 q 已学会真实质量**。S4/S5 的 `a_in` 已混入 IR，q4/q5 更不能简单称作“纯 RGB 图像质量”。

训练时 backbone 返回主 IR 特征、辅助融合特征和质量统计。`RTv4_Infranet` 分别执行：

```text
IR features    → 同一 encoder → 同一 decoder → main outputs
fusion features→ 同一 encoder → 同一 decoder → aux outputs
```

两次前向共享参数，损失为：

```text
L_total = L_main_detection + 0.25 · L_aux_detection
```

0.25 在 criterion 包装中写死，不是任意 YAML loss weight 会自动覆盖的系数。**这叫共享头辅助检测训练，不是已经实现了显式蒸馏。** 当前没有主/辅 soft logits、特征、方向分布之间的 teacher-student 对齐损失，也没有质量标定真值损失。

推理有两种模式：

- `ir`：跳过 RGB/辅助融合路径的执行，用主 IR 特征检测。
- `rgb_ir`：执行 RGB stem 与质量门，使用融合输出检测。

IR-only 模式并不自动删除 RGB/QualGate 模块或其权重，所以**总参数量、实际执行参数量、checkpoint 大小**是不同指标。当前 ATR InfraNet YAML 设置的是 `rgb_ir`，若研究 IR-only 部署必须显式区分。

同一个含 BN 的 encoder 在一次训练迭代中处理两种特征分布，运行均值/方差会更新两次。是否需要分路 BN、冻结 BN 或其他归一化，应通过公平实验判断，不能本次擅自修改。

### 7.6 HybridEncoder：把三尺度特征变成检测记忆

`HybridEncoder` 先把不同输入通道投影到 hidden_dim，再做尺度内和跨尺度处理：

1. `input_proj`：用 1×1 卷积等对齐各尺度通道。
2. AIFI/Transformer：默认只对 S5（`use_encoder_idx=[2]`）执行尺度内自注意力，降低高分辨率注意力代价。
3. FPN：自顶向下上采样并融合浅层定位细节。
4. PAN：自底向上进一步聚合多尺度信息。

输出仍是三张特征图，而不是已经完成类别/框预测。它们随后被解码器展平为多尺度 memory。

部分卷积结构支持部署重参数化，把训练多分支折叠为推理结构；这与“删除 InfraNet 辅助分支”不是一回事。

本文件还保留 teacher 蒸馏用的投影接口，但当前 OBB 配置没有启用。要启用 DINO teacher，需额外选择三通道模态、处理 patch 网格，并给 OBB criterion 配套损失，不能只加一个 YAML 开关。

二维位置编码的 W/H 遍历顺序与 NCHW flatten 顺序不一致，已用小尺寸示例确认；具体证据见第 8.2 节。

### 7.7 OBB 解码器：从候选到五维旋转框

`RTDETRTransformerv2OBB` 的主要流程如下：

1. 展平 encoder 的三尺度输出，保留各层的 H/W 信息，形成多尺度 memory。
2. 对 memory 位置产生编码器候选类别/框预测，按分数选 top-k 普通查询。
3. 初始化查询内容和参考 OBB；当前 `learn_query_content=False` 时初始内容来自选中的 memory，并有 detach 处理。
4. 训练时拼入 GT 派生的去噪查询，添加对应注意力 mask。
5. 每层执行查询自注意力、旋转可变形跨尺度注意力和 FFN，再更新类别及五维框。
6. 训练输出普通最终预测、辅助层、编码器候选和 DN 信息；验证只保留选定解码层的正常预测。

旋转可变形注意力并不是先把整幅特征图旋转。每个 head 在参考框局部坐标中预测若干偏移，按框大小缩放、按角度旋转，然后映射到各尺度的 `grid_sample` 坐标，按学习到的权重聚合采样值。

典型 refinement 可以概括为：

```text
box_next = sigmoid(delta + inverse_sigmoid(box_reference))
```

这里第五维角度也在 sigmoid/logit 域中更新。**角度 logit 为 0 对应归一化角度 0.5，即 90°，不是 0°。** 不能把源码中角度偏置初始化为 zero 的文字直接解释为零度朝向。

该检测头已经有空间可变形注意力，但它作用在**融合后的 memory**，不自动提供 IR 与 RGB 之间的显式对应关系。想研究跨模态对齐，应区分“检测头的采样”与“融合阶段的跨模态采样”。

当前非正方形归一化旋转公式存在误差；这会影响对新融合模块收益的判断，详见第 8.1 节。

### 7.8 去噪查询：加速训练，不是测试时使用标签

`get_contrastive_denoising_training_group_obb` 根据 batch 最大 GT 数生成填充后的组，包含正负样本、类别扰动和 xywh 扰动，并返回匹配索引及分组元信息。

- 正样本学习恢复真值；负样本主要提供更困难的分类监督。
- 普通查询不能读取带 GT 信息的 DN 查询，不同 DN 组也有屏蔽关系。
- DN 输出在 loss 中分开监督，不作为正常评估预测。
- 当前角度没有加入去噪扰动，不能声称已实现“旋转角去噪”。
- `num_denoising=100` 是分组预算参数，实际数量还受最大 GT 数、正负复制和分组取整影响。

研究角度噪声时需考虑半周周期与近正方形目标的方向不可辨识性，不能简单给归一化角度加普通高斯噪声再截断。

### 7.9 Hungarian 匹配、检测损失与条件损失

**训练匹配**先在 `HungarianMatcherOBB` 中计算成本矩阵，再在 CPU 上调用 SciPy 最优指派。指派是离散结果，不是要反向传播的可学习模块。

概念成本为：

```text
C = 2 · C_class + 0 · C_bbox + 5 · C_corner_Hausdorff + 5 · C_KLD
```

这是当前主要配置的权重，不是所有实验的普遍公式。`C_bbox` 为五维 L1，但当前系数为 0；名为 `cost_giou` 的项调用 OBB KLD，并非旋转 GIoU。Hausdorff 基于四个角点集合，不是整个矩形边界的精确连续 Hausdorff。

**训练损失**的基本检测部分为：

```text
L_detection = 1 · L_focal + 5 · L_l1 + 5 · L_kld
```

还会根据输出结构加入 decoder 辅助层、encoder、DN 等项，所以最终 loss 不是只有这三个标量相加。

- focal 使用未匹配查询作背景，处理类别不平衡。
- L1 对五维归一化框回归，包括角度，当前不识别 0 与 1 的半周等价性。
- KLD 将框转换为二维 Gaussian，利用均值和协方差刻画位置、大小与方向；它不是评估 IoU 的替代定义。
- `use_uni_set=True` 聚合多层匹配对供回归使用，可能让多个 query 回归同一个 GT；分类仍用各层自身指派，不等于取消所有一对一约束。
- criterion 当前假定存在 `aux_outputs`。直接关闭 decoder 的 `aux_loss` 不是无需配套修改的安全设置。
- NaN loss 会被归零，可能把几何/数值问题隐藏成“训练还能跑”；应主动记录并定位发生位置。

**条件监督**只在对应分支与配置启用时计算：

```text
L_condition_raw = SmoothL1(height) + SmoothL1(angle) + SmoothL1(time)
                + 0.5 · [CE(weather) + CE(illumination) + CE(location)]
L_condition = 0.5 · L_condition_raw
```

第二个 0.5 来自 prototype 的外部 `loss_cond` 权重。因此三组分类条件相对于回归条件还多一次内部权重，不应将六任务称为完全等权。

这里的 angle 监督是 UAV 观测角；检测框角度在 `loss_l1/loss_kld` 中处理。缺失条件也按默认值监督会引入伪标签，需与真正物理先验的价值分开讨论。

### 7.10 后处理和本地 AP：训练成本不是评估指标

`PostProcessorOBB` 对每个 query 的类别 sigmoid 分数展平，从 `query×class` 中取 top-k，再将框还原到原图像像素单位并将角度乘 π 转为弧度。

- 不默认做 NMS，也不默认做置信度阈值过滤。
- 一个 query 可以在不同类别上被选中；top-k 并不严格等于 300 个互不重复的目标。
- 普通路径按原图尺寸缩放；设置 `input_shape` 的路径假设等比例缩放和右/下补边，没有居中 padding 偏移反变换。
- 训练匹配使用 Hungarian；验证 TP/FP 使用按分数排序后的类别内 IoU 匹配。这是两个不同过程。

`DotaEvaluator` 虽叫 DOTA，也可用于 ATR/DroneVehicle，只要 dataset 提供它需要的类别和真值接口。默认仅用 IoU=0.5，通过 mmeval 的 VOC2007 十一点方式计算 AP，所以默认 `mAP` 应解释为 **旋转 AP50，VOC2007 插值方式**。

要请求一组 IoU 阈值，可以在运行参数中增加：

```powershell
-u "evaluator.iou_thrs=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]"
```

这只是需追加到完整命令中的参数片段。当前多阈值统计还使用 0.5/0.75 的特定汇总项；应保留这两个阈值。`AP50_95` 是**实际传入阈值列表的平均**，不能只传 `[0.5,0.75]` 就宣称测了完整 AP50:95；即使传入完整列表，其 AP 插值方式仍不能直接等同 COCO 官方评估。

mmeval 中没有 GT 的类别会从均值中排除；有 GT 但没有预测的类别 AP 为 0。`scale_ranges` 在当前实现中没有真正应用尺寸区间过滤，不能直接据此报告已实现的 AP_small/AP_medium/AP_large。

### 7.11 Solver、EMA、调度器与模型统计

`BaseSolver` 管组件与状态，`DetSolver` 管 epoch 和阶段，`det_engine` 管每个 batch。训练流程包括前向、加权 loss、梯度缩放/反向、裁剪、optimizer step、EMA 和日志；验证通常使用启用的 EMA 模型。

FlatCosine scheduler 由 `lrsheduler` 触发，根据 loader 长度将 warmup/flat/no-aug 转为迭代时间。它与保留的 MultiStepLR 分支不同，比较实验时要核对实际执行的是哪一种。

`ModelEMA` 保存平滑后的模型参数和缓冲区。当前 `decay_fn` 的 lambda 捕获了构造时的 decay，因此后续只改 `self.decay` 不会改变真正的衰减系数；这会影响阶段切换“重置 EMA 衰减”的预期。

`det_solver.py` 的第二阶段逻辑有重新载入最佳状态的路径，可能改变模型、优化器和 epoch 相关状态。ATR 的 stop_epoch 等于训练总轮数时不会触发该阶段，不能因此证明其他长训练配置的阶段切换安全。

`profiler_utils.stats` 深拷贝部署态模型并按配置尺寸估算运算。它提供的是所追踪 dummy 路径的估算，不是完整训练耗时，也不是实测 FPS；InfraNet 的未执行辅助模块仍可能被总参数统计计入。论文应额外报告实际模式、实测延迟和峰值显存。

<a id="audit"></a>
## 8. 代码审查：问题、证据与风险

以下区分**实际小张量验证**、**源码/配置可直接确认**和**仍需专门实验验证**，避免把推测写成已证实结论。本次没有修复任何可执行代码。

### 8.1 已验证：矩形输入下旋转采样使用了不一致的度量

相关位置：`engine/rtv4/rtdetrv2_obb_decoder.py` 的 `MSDeformableAttention.forward`。

设图像宽高为 W/H，`D=diag(W,H)`，旋转矩阵为 `R(theta)`。局部偏移 v 已按 `w/W`、`h/H` 缩放，因此 v 是归一化坐标中的局部轴向偏移。

当前实现相当于：

```text
offset_normalized_current = R(theta) · v
```

若 theta 表示真实图像像素坐标中的旋转角，应该先恢复像素度量、旋转，再归一化：

```text
offset_normalized_consistent = D^(-1) · R(theta) · D · v

dx = cos(theta) · vx - sin(theta) · (H/W) · vy
dy = sin(theta) · (W/H) · vx + cos(theta) · vy
```

W=H 时 D 与 R 可交换，两个公式相同；矩形输入通常不相同。这不是张量 shape 问题，而是 x/y 度量不一致。

用实际源码 AST 提取的注意力类做独立 CPU 验证，设置一头一点、参考中心 `(0.5,0.5)`、框宽 128 像素、高 64 像素、旋转 90°，局部 x 方向偏移尺度为 0.5：

| 条件 | 预期像素偏移 | 当前实现像素偏移 |
| --- | --- | --- |
| W=640，H=512 | `(0,64)` | 约 `(0,51.200012)` |

该误差与 512/640 的比例对应，恰好涉及当前 ATR 的输入尺寸。**这证明局部采样公式不符合像素旋转语义，不证明修正后 AP 一定提高，也不表示已定位所有训练问题。**

同类检查还应覆盖 `utils_obb.py`：

- 像素 OBB 的 Gaussian 协方差 `Sigma_px` 在归一化坐标中应变为 `D^(-1) Sigma_px D^(-T)`，均值也需同步变换。
- 将归一化 w/h 直接当作等距旋转矩形构造协方差，与上述完整变换一般不等价。
- KL 的仿射不变性要求两分布的均值、完整协方差一致变换，不能只变 w/h 就援引该性质。
- 四角点构造与 Hausdorff 也需明确是在像素度量还是另一种归一化度量下比较；如果声称评价真实像素框，须保持角点变换的一致性。

**建议的后续修复顺序**：统一框表示/坐标度量→采样公式→协方差/角点→resize 与反变换→用已知几何做单元测试。修复本身是正确性工作；要成为理论贡献，还需要新的性质、建模或泛化证据。

### 8.2 已验证：二维位置编码与特征 flatten 的位置顺序不一致

相关位置：`engine/rtv4/hybrid_encoder.py` 的 `build_2d_sincos_position_embedding`。

当前用 W/H 的 `meshgrid(..., indexing='ij')` 再展平；NCHW 特征的 `flatten(2)` 则按 H 行、W 列展平。用 W=3、H=2 得到：

```text
位置编码对应坐标：
(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)

NCHW 特征 flatten 对应坐标：
(0,0), (1,0), (2,0), (0,1), (1,1), (2,1)
```

两个 Tensor 长度相同，所以普通 shape 检查会通过，但同一索引对应了不同空间位置。正方形网格也可能表现为转置式关联，并非“正方形一定正确”。

后续应在构造网格、缓存位置编码和实际 feature flatten 之间统一索引约定，并覆盖矩形和正方形样例。本次仅增加说明注释，没有替换公式。

### 8.3 其他审查发现

优先级含义：P0 为正式复现/报告前必须澄清的启动或可信度问题；P1 为训练与建模正确性风险；P2 为扩展、效率和维护风险。

| 优先级 | 位置/问题 | 证据状态与影响 | 后续处理建议 |
| --- | --- | --- | --- |
| P0 | `requirements.txt` 三个 `#` 紧贴包名 | 标准 requirement 解析实际失败；直接批量安装不可靠 | 独立修正依赖格式、建立完整版本记录；本次不修改 |
| P0 | TSTI fast 配置 | 全源码没有 `AsymHGNetv2_TSTI_Sup`，OBB loss 映射没有 saliency | 视为未完成配置，不作为可运行分支或已有创新 |
| P0 | ATR test 用于 val、Drone test 用于 train | 配置可直接确认 | 明确重划分协议；禁止测试集选择模型后又当独立测试 |
| P0 | 读图异常后随机换样本 | 数据集重试逻辑可确认；val 也可能重复/缺失 ID | 离线检查坏数据，评估记录唯一 ID 和应评/实评数量 |
| P0 | `dota_auto_evaluator.py` 的 mAP 占位 | 返回 epoch 作为 `stats['mAP']`，不是真实准确率 | 不用于本地精度结论；与真正远程评分分离 |
| P1 | 五维 L1 的角度非周期 | 源码直接对第五维作绝对差；0/1 半周等价被当作远距离 | 研究周期表示及近正方形方向弱可辨识处理 |
| P1 | 条件缺失/越界默认值仍被监督 | ATR 解析和 criterion 可确认 | 添加有效性掩码、报告缺失率、单独做有/无伪标签消融 |
| P1 | q 无质量校准目标 | QualGate 只有检测梯度监督；q4/q5 输入已含 IR | 不能把 q 解释为真实 SNR/照度，需设计任务相关质量标定 |
| P1 | 共享 encoder BN 两次更新 | InfraNet 容器执行两次共享 encoder 前向 | 比较共享/分路/冻结归一化，记录两种部署模式 |
| P1 | NaN loss 归零 | OBB criterion 可确认 | 增加发生次数与上下文记录，不能用归零掩盖数值错误 |
| P1 | AMP `device_type=str(device)` | 代码可确认；带索引设备字符串不是预期类型参数 | 使用默认设备/可见 GPU 选择；后续统一为 device 的类型 |
| P1 | stage-2 reload 与 EMA decay 更新 | 最佳状态重新加载及 lambda 捕获可确认 | 分别验证阶段转换、resume、一致的 scheduler/EMA 状态 |
| P1 | 非零 affine 旋转的图像/标签方向 | 标签使用 OpenCV 旋转矩阵，图像使用 torchvision affine；接口正角方向不同，**仍需实际图像复现确认** | 用非对称标记图+OBB 检查正负角；当前 rotation_range=0 未触发 |
| P1 | `utils.py` 的 discrete 采样边界 | 离散分支用 h-1 同时限制 x/y，矩形特征的 x 应按 w-1；源码可确认，当前使用 default 分支 | 切换 discrete 前单独验证矩形边界，不能认为 default 的运行已覆盖它 |
| P1 | 后处理缩放/padding 假设 | `input_shape` 分支仅覆盖等比缩放和右/下补边 | 保存真实几何变换元信息，验证 stretch/居中 padding 的反变换 |
| P1 | 不可信 YAML/checkpoint 反序列化 | 配置读取使用 `yaml.Loader`，部分 checkpoint 使用 `torch.load`/pickle | 只加载可信文件；本次独立检查用 safe_load 不代表运行加载器已改为安全模式 |
| P2 | 缺少本地预训练时的分布式调用 | HGNetv2 下载/错误处理直接调用 rank/barrier | 单进程提供本地权重或明确关闭预训练，后续增加初始化检查 |
| P2 | 多机 GPU 索引 | distributed 工具将 global rank 与本机设备编号混用 | 另行验证 LOCAL_RANK 逻辑；当前只建议单机示例 |
| P2 | Compose 独立 default policy | 日志读取 epoch；当前继承配置有该键，手工最小 policy 可能没有 | 区分当前有效配置与独立构造，不误报现有 ATR 启动失败 |
| P2 | `aux_loss=False` 与 criterion 假设 | OBB criterion 要求 aux_outputs | 修改解码层/辅助输出时必须配套 loss，而非只改一个开关 |
| P2 | teacher 的输入与网格 | teacher 预期三通道，DINOv3 有方形 token 网格假设；OBB 没有完整蒸馏映射 | 做模态选择、矩形 patch 适配和 OBB 专用蒸馏联调 |
| P2 | FLOPs/参数与真实部署模式 | profiler dummy 复制三通道，总参数包括未执行模块 | 区分总参数/活跃参数/实际延迟，不声称已测 FPS |
| P2 | `scale_ranges` 与 AP 名称 | mmeval 未真正按尺度区间筛选；多 IoU 平均取实际阈值列表 | 自行实现并验证尺寸评估，明确 AP 插值和阈值 |

`DotaAutoEvaluator`/Offline 会删除并重建选定的结果目录。若未来启用，必须使用专门空的实验输出路径，不要指向数据集、项目根目录或重要已有文件夹。远程提交还应确认数据与标注结果是否允许外传，不在配置中公开真实 API key。

### 8.4 本次审查不能证明的事情

- 没有完整训练/数据集评估，不能报告“修复带来了多少 AP”。
- 小张量和梯度检查不能证明模型更鲁棒、q 有物理意义或存在学术新颖性。
- 没有核实源码注释中的 InfraNet 论文链接及结构对应关系，不能称为官方实现或忠实复现。
- 没有进行系统文献检索，不能断言任何建议“前人从未提出”。
- 没有修改上述问题，因此运行代码仍会保留对应行为。

<a id="existing-results"></a>
## 9. 已有实验记录如何解读

以下数值仅摘录自当前 `experiment.md`，**不是本次重跑或独立验证的结果**。按当前默认本地评估器，应优先将其中 mAP 理解为 AP50；历史运行若改过 evaluator，则以完整原始日志为准。

| 实验 | 笔记中的最佳 epoch | 记录 mAP | 百分数显示 |
| --- | --- | --- | --- |
| Single IR | 20 | 0.44100 | 44.10% |
| Concat | 22 | 0.53416 | 53.42% |
| Add | 20 | 0.54038 | 54.04% |
| LGDense prototype | 23 | 0.52123 | 52.12% |
| LGDense self-driven | 17 | 0.51557 | 51.56% |
| InfraNet | 未记录 | 未记录 | 不能补造数值 |

在这些单次记录中：

- Add 比 Single IR 高 **9.938 个百分点**，说明这个记录里增加 RGB 有明显潜在价值。
- prototype 比 Add 低 **1.915 个百分点**，当前复杂门控还没有超过简单基线。
- prototype 比 self-driven 高 **0.566 个百分点**，不能从这组数值断言“显式条件先验完全无效”。

但笔记没有给出全部 seed、方差、完整日志、代码版本、权重来源和清晰数据协议。当前 YAML 相同也不证明历史训练使用了完全一致设置。上述差值不能直接解释为统计显著或普适因果结论。

关于原笔记的两个判断，应更谨慎：

1. “两模态没有互动”：LGDense 有联合 z 带来的间接条件耦合，只是没有显式局部对应交互。
2. “引入显式先验失败”：只能说这组记录没有超过 Add；伪标签、loss 权重、地点捷径、梯度冲突、训练强度和随机波动都尚未排除。

`experiment.md` 中还出现当前不存在的 Drone IR YAML 路径。不能直接复制所有旧命令；应以第 5 节列出的真实配置为准。

<a id="research"></a>
## 10. 能否创新：理论与代码实验路线

**可以在本项目上做创新，但“能改出新模块”“能提高某次 AP”“具有可发表的新颖性”是三件不同的事。** 当前最有价值的基础是完整的双模态 OBB 主链路、简单融合对照、条件多任务分支和 IR-only/RGB-IR 两种计算路径；最需要先处理的是几何一致性和实验协议。

以下是研究建议，不是已实现功能或保证有效的方案。真正实施涉及可执行代码、配置和实验协议，需要在本次“仅注释”工作之外另行开展。

### 10.1 理论创新与工程实验应如何连接

| 类型 | 应交付什么 | 本项目能提供什么 | 不能冒充什么 |
| --- | --- | --- | --- |
| 纯机理/理论研究 | 明确假设、推导、可识别条件、性质/界、反例及相对既有理论的新内容 | 旋转坐标度量、模态噪声/相关误差、方向可辨识性、质量与任务收益的机制问题 | 把已知加权平均、注意力或周期编码换名，不自动成为理论创新 |
| 算法与实验研究 | 可复现方法、统一基线、充分消融、稳定收益、代价和失败场景 | YAML 装配、多种融合基线、OBB loss/eval、可插入的 backbone/decoder | 一次最佳 seed 或多加参数后的涨点，不自动说明有效机制 |
| 物理成像机理研究 | 传感器/辐射假设、标定与观测量、可检验的成像解释 | 高度/天气/光照等粗条件，可辅助分组与提出假设 | 当前 uint8 IR 三通道不等于绝对温度、辐射定标或真实 SNR |

若从红外辐射机理出发，温度、发射率、反射辐射、大气透过率、响应曲线与自动增益都可能影响观测。当前加载器只读图像和粗条件，没有完整传感器标定数据；因此不能直接把灰度代入普朗克定律就宣称获得了可辨识的真实温度或可靠质量。

理论工作可以先建立带噪声/偏差的简化观测模型，但必须说明哪些量可观测、哪些参数无法由当前数据唯一恢复。代码实验用于检验假设，而不是替代假设的物理依据。

### 10.2 方向一：度量一致与方向可辨识的旋转检测

**已有缺口**：旋转采样、Gaussian 几何和角点可能使用不同度量；角度 L1 不周期；近正方形目标对方向的辨识较弱。

**可研究内容**：

1. 统一像素框、归一化坐标、采样点与协方差的变换关系，证明在明确的坐标换单位下几何计算一致。
2. 使用半周周期的方向表示，例如 `(cos(2theta), sin(2theta))`，避免 0/π 断点。
3. 根据目标的长宽差刻画方向可辨识性，而不是对正方形和细长车辆施加同等角度惩罚。
4. 让 matcher、回归 loss、DN 角度噪声和 refinement 的表示一致，而不是只替换一个 loss。

一个用于分析的形状方向向量为：

```text
rho_signed = (w² - h²) / (w² + h² + epsilon)
p_shape = rho_signed · [cos(2theta), sin(2theta)]
```

它具有两个容易验证的性质：

- theta 加 π 时向量不变，符合旋转矩形半周周期。
- 同时交换 w/h、让 theta 加 π/2 时，rho 与三角项都变号，向量仍不变。

接近正方形时 rho 接近 0，方向项自然变弱。另一种简单候选是用**真值**各向异性权重乘 `1-cos(2Δtheta)`，避免预测模型通过把框做成正方形来逃避方向损失。

这些表示与 Gaussian/协方差关系属于已知几何工具，**单独采用不保证新颖**。潜在贡献在于发现当前耦合误差、建立完整的度量一致/不确定性模型，并证明与检测训练的具体关系。

还需区分“坐标换单位”与“实际非等比缩放图像”：一般旋转矩形经真实非等比缩放会变成平行四边形，OBB 表示并不对所有仿射变换封闭，不能过度宣称整个检测器具有任意仿射精确等变性。

**实验设计**：矩形/正方形输入、0/45/90 度、长条/近方形目标、跨角度边界；分别只修采样、只修位置编码、两者都修，再比较新几何建模。所有融合基线共享修复，防止新方法独享 bug fix。

**后续可能涉及的文件**：`rtdetrv2_obb_decoder.py`、`utils_obb.py`、`obb_matcher.py`、`rtv4_obb_criterion.py`、`denoising_obb.py`、`transforms_obb.py`、`postprocessor.py`。

### 10.3 方向二：任务收益校准的局部融合与跨模态对应

**已有缺口**：LGDense 只有间接全局耦合和 RGB 局部门；QualGate 的 q 为图像级代理且没有真实质量监督；两者均未显式解决局部错位。

#### A. 先建立“质量”与误差的理论关系

假设两模态特征是在同一坐标、同一尺度上对相同潜变量的无偏估计，误差方差为 `v_ir`、`v_rgb`，协方差为 c。以 w 表示 IR 权重：

```text
F(w) = w · F_ir + (1-w) · F_rgb
Var(error) = w² v_ir + (1-w)² v_rgb + 2w(1-w)c

w_ir_opt = clip((v_rgb - c) / (v_ir + v_rgb - 2c), 0, 1)
```

该公式要求分母为正，并假设对齐、无偏及可比较特征。它说明最优权重取决于**两流误差及相关性**，不是只由 RGB 亮度决定。实际深特征可能有偏、有错位、尺度不一致，因此不能把这个已知最小方差公式直接当作当前网络的正确性证明。

潜在理论方向是进一步建模局部错位、共同噪声和任务相关偏差，推导在什么条件下 RGB 信息值得引入，以及何时应退回 IR。

#### B. 显式局部对应，而不只通道加权

一个受算力约束的候选形式为：

```text
delta_i(p) = Σ_k attention_i,k(p) · V_rgb_i(p + offset_i,k(p))
F_i(p) = I_i(p) + gate_i(p) · Phi_i(I_i(p), delta_i(p))
```

让 IR 内容提出 query，RGB 提供 key/value 或稀疏候选；偏移用于学习局部对应，gate 结合两个模态及对应不确定性控制增量。优先在低分辨率尺度或少量采样点上试验，避免全分辨率二次方注意力成本。

显式 IR 残差只提供结构锚点，不保证 AP 不降。若**固定查询选择等离散步骤**并假设局部 decoder 为 L-Lipschitz，只能得到：

```text
||D(F_ir + gate·delta) - D(F_ir)|| <= L · ||gate·delta||
```

这约束预测扰动，不是“检测 AP 永不下降”的证明；top-k、匹配和 AP 本身还包含离散变化。

#### C. 让可靠性预测“检测收益”，而不冒充照度/SNR

可以定义训练时的反事实辅助目标：

```text
utility_target = sigmoid((Loss_IR - Loss_reference_fusion) / temperature)
```

其中两项应在同一 GT/可比较指派上计算，target 需 detach，或来自冻结/EMA teacher。不能让门控通过操纵自己的参考 loss“刷高质量分数”。推理不需要真值，也不能使用 test 标注生成门控。

如果质量定义为“融合是否降低损失”的事件，可检查 Brier/ECE 等校准；若定义为连续收益，应报告收益误差、排序相关性和分组性能。**任何 sigmoid 输出都不是天然经过校准的概率。**

**关键消融**：Add、Concat、参数匹配卷积、仅局部 gate、仅对应采样、仅质量监督、完整方法；同时测清洁、黑夜、遮挡、噪声和错位。

**后续可能涉及的文件**：`dual_hgnetv2_lg_dense.py`、`qual_gate.py`、`dual_hgnetv2_infranet.py`、模型容器和 OBB criterion，以及新的成对扰动/诊断脚本。

### 10.4 方向三：把双模态训练真正转化为 IR-only 部署收益

**已有缺口**：InfraNet 当前只有共享头的主/辅助检测，并没有显式蒸馏；IR-only 还保留未执行的 RGB/门控参数。

候选方案是使用 RGB-IR 模型作为训练特权 teacher、IR 模型作为部署 student：

```text
L_student = L_IR_detection
          + lambda_cls · L_soft_class
          + lambda_geo · L_orientation_geometry
          + lambda_feat · L_foreground_feature
```

- teacher 应冻结或以明确 EMA 规则更新，不能把共享头辅助分支自动称作 teacher。
- 蒸馏只在前景/可靠对应位置增强，避免把坏 RGB 和背景噪声传给 IR。
- 角度蒸馏使用周期、各向异性或协方差表示，不直接在 0/1 边界蒸馏普通角度 L1。
- teacher/student 查询不同，需要基于 GT、共享指派或明确的 query 对齐策略比较输出。
- 分离归一化统计，避免双路共享 BN 的混合统计成为隐性变量。
- 真正部署时构建只含 IR 的网络并提取对应 state，不只设置 `eval_mode='ir'`。

**关键对照**：同初始化的 Single IR、当前 InfraNet 的 IR 模式、普通类别/特征蒸馏、方向/质量感知蒸馏；IR-only student 在相同算力下是否稳定获益，并单独报告 teacher 的训练代价。

多模态特权学习/蒸馏已有大量研究。潜在新内容必须具体到“方向可辨识性、质量与对应关系如何控制蒸馏”，而不是“训练用 RGB、测试用 IR”这个概念本身。

**后续可能涉及的文件**：`rtv4_infranet.py`、`rtv4_obb_criterion_infranet.py`、IR 骨干/部署包装、teacher 接口和 checkpoint 导出工具；当前 DINO teacher 不能直接当作已就绪的 OBB 双模态 teacher。

### 10.5 方向四：重新检验六组条件先验，而非凭一组结果放弃

高度、观测角、光照、天气等可以影响目标尺度、模态可靠性和成像退化；location 也可能变成类别/场景捷径。关键是区分“对成像有因果解释的条件”和“只在当前划分里相关的标签”。

建议按以下顺序研究：

1. 先加入条件有效性掩码，让缺失/默认值不参与伪监督，记录每任务有效样本量。
2. 比较无辅助头、真实条件、打乱条件、随机等容量辅助任务，分清物理信息与普通多任务正则化。
3. 单独监督高度/角度/时间、单独监督分类条件、去掉 location，定位有效/有害任务。
4. 测量检测 loss 与各条件 loss 对共享参数的梯度夹角，检查任务冲突，再讨论权重自适应或训练分阶段。
5. 做 leave-location-out、leave-weather-out 或视频/场景外验证，观察是否依赖场景捷径。
6. 比较“仅监督 z”“z 控制全局门”“z 加局部对应”的收益，判断条件是否真正改变融合决策。

可提出条件化误差/噪声模型，但要承认这些标签往往相关、缺失且粒度粗，不能直接证明因果性。梯度平衡、辅助任务和条件编码本身也不是新概念；潜在贡献在于可检验的成像解释和跨条件泛化。

**后续可能涉及的文件**：`atrumod_dataset.py`、`dual_hgnetv2_lg_dense.py`、`rtv4_obb_criterion.py`、数据划分/诊断工具及实验 YAML。

### 10.6 方向五：真实节省计算的模态/查询路由

如果目标是实时部署，可以研究：

- 仅在 IR 不确定或对应可靠时执行 RGB 的部分计算。
- 依据场景/目标复杂度分配 query 数或提前停止解码层。
- 在小目标密集图像与简单图像间动态分配计算预算。

但**计算完 RGB 特征后乘一个零，并没有节省 RGB backbone 的运行时间**。真实路由需要改变执行图，而且 gate 自身、batch 分组和内核调度也有代价。

理论上可以建立“期望检测风险 + 计算预算”目标，实验必须测实际跳过率、质量-延迟曲线和硬件收益。当前 InfraNet 的 RGB 分支本身已较轻，额外路由是否划算更不能凭 FLOPs 推断。

动态路由、动态 query 和 early-exit 已有研究，需要明确新增的是何种可靠性建模、旋转检测机制或部署约束，而不是只改 `num_queries`。

### 10.7 必做的对照实验与报告方式

#### A. 统一训练控制

- 至少固定 seed 0/1/2，条件允许时增加到 5 个；报告均值、标准差和逐 seed 结果，而不只挑最好值。
- 固定 train/val/test、视频/场景分组、输入尺寸、全局 batch、优化器、学习率、预热、训练迭代量、AMP、EMA、增强和预训练来源。
- 采用相同验证集选择规则；test 只用于最终评价，不能用它选择 loss 权重、epoch 或架构。
- 几何/位置编码等正确性修复对全部方法一致启用，并保留原始实现对照记录。
- 同一类比较使用相同推理模态；IR-only 与 RGB-IR 不应在表中混为一个算力条件。
- 对新增模块加入参数/计算匹配的普通卷积基线，排除“只是多了容量”。

#### B. 最小方法矩阵

| 对照组 | 要回答的问题 |
| --- | --- |
| Single IR / Add / Concat | 原始双模态收益有多大，简单融合基线是否已经很强 |
| Add + 等容量普通融合层 | 新模块是否只是参数量/非线性更大 |
| LGDense 无头 / 有真实条件 / 条件打乱 | 条件语义、监督正则与融合结构各贡献多少 |
| 仅局部门 / 仅对应采样 / 仅质量目标 / 完整方法 | 质量、对齐和空间交互是否各有独立收益 |
| InfraNet IR / InfraNet RGB-IR | 共享训练对 IR 主路的帮助、融合推理的代价 |
| 普通蒸馏 / 方向质量感知蒸馏 | 特权训练收益是否超出通用蒸馏 |

这些都是未来实验设计，不表示当前已有相应配置或结果。优先分阶段做，不必一开始穷举所有组合。

#### C. 指标不只看 AP50

| 指标 | 报告要求 |
| --- | --- |
| AP50、AP75、多 IoU 平均 | 明确是旋转 IoU、实际阈值列表、AP 插值方式；当前默认不是 COCO 评估 |
| 各类 AP | 报告每类 GT 数，防止类别缺失或均值规则改变影响结论 |
| 尺度/长宽比分组 | 在原图 GT 像素定义上明确阈值；当前 scale_ranges 未实现，需新实现与测试 |
| 条件分组 | 按高度、观测角、时间、光照、天气、地点统计，报告有效样本量与缺失组 |
| 几何诊断 | 中心/长宽/周期角度误差、近方形与细长目标、采样点/角点可视化 |
| 可靠性诊断 | q/gate 与实际融合收益的相关性、校准、错位/退化情况下的变化 |
| 效率 | 总参数、活跃参数、FLOPs/MACs、实际延迟/FPS、峰值显存、checkpoint 大小 |

AP75 与几何诊断对定位/方向改进往往比只看 AP50 更有信息，但仍需要分组和误检/漏检分析。

#### D. 鲁棒性与反事实实验

1. RGB 曝光降低/过曝、模糊、噪声、遮挡与压缩退化，分多个强度；IR 也做独立退化作为对照。
2. RGB-only 平移若干像素或小角度旋转，GT 始终锚定 IR，评估局部错位耐受；若是整场景几何增强，则必须同步变换两模态与 GT。
3. RGB dropout/置零，与正常双模态比较。**不要通过删除 RGB 文件模拟缺失**，当前随机重试会换掉评估样本。
4. 清洁/退化同一帧配对评价，检查新增 RGB 是否在坏条件下造成负增益。
5. 在真实夜间/天气分组上验证，不能把合成退化等同于全部真实分布变化。
6. 质量目标与 shuffle/constant gate 比较，判断 gate 是否学到有意义的选择，而非固定缩放。

#### E. 实测速度与统计可信度

GPU 计时应预热，使用 CUDA event 或同步计时，固定 batch、输入、AMP、部署模式和后处理。分别报告纯模型延迟与包含数据搬运/后处理的端到端延迟，评估器 IoU/AP 耗时不算日常推理延迟。

至少在 batch=1 与实际部署 batch 下测量，报告预热/重复次数及 GPU 型号；推理关闭梯度。calflops 对特殊采样/路由操作的支持需核对，FLOPs 统计规则也要统一。

配对 seed 的差值比单次最佳值更有解释力；数据来自视频时，还应考虑帧间相关性，必要时按视频/场景做区间估计。少量 seed 的波动不能凭一个小差值宣布显著。

### 10.8 推荐实施顺序与代码落点

| 顺序 | 工作 | 对应现有代码 | 验收标准 |
| --- | --- | --- | --- |
| 1 | 数据/依赖/评估协议整理 | dataset YAML、加载器、requirements、本地 evaluator | 可重复读数据，唯一 ID 数量正确，test 不参与调参 |
| 2 | 正确性基线 | OBB decoder、HybridEncoder、utils_obb、变换/后处理 | 已知几何通过，原始与统一修复对照可复现 |
| 3 | 稳定复现 IR/Add/Concat | 三种骨干与同一训练链 | 多 seed 均值/方差、完整日志与效率表 |
| 4 | 一条主创新：局部对应 + 任务可靠性 | 融合骨干、门控、criterion | 参数匹配下稳定收益；错位/退化实验支持机制 |
| 5 | 或另一条主创新：IR-only 特权蒸馏 | InfraNet 容器/criterion、独立 IR student | 同 IR 推理算力下获益，部署参数确实只含 IR |
| 6 | 条件先验与动态计算扩展 | ATR 条件解析、LGDense、调度/路由 | 有效性掩码、OOD 分组与实际硬件收益 |

不建议一开始同时叠加条件、对齐、蒸馏、动态路由和新 decoder。先提出一个可证伪的核心假设，再用最小实现和明确消融证明它。

### 10.9 对“能不能做出创新”的最终判断

- **理论上有研究空间**：坐标度量、方向不确定性、模态相关误差和任务收益可靠性，均能形成明确推导与反例。
- **代码实验上有足够落点**：融合骨干、OBB 几何损失、辅助训练与推理模式已经具备改造接口。
- **当前最值得优先尝试**：统一几何正确性后，研究带对应关系的局部可靠性融合；若目标明确是 IR-only 部署，则优先做方向/质量感知特权蒸馏。
- **不能保证**：涨点、超过 SOTA、物理解释完全成立或论文录用。需要系统文献对比、干净实验协议和可复现实证。
- **本次没有实现这些方法**：仅完成代码说明、注释和研究设计，保留原始运行逻辑。

<a id="verification"></a>
## 11. 验证、复现与后续工作清单

### 11.1 本次补充注释的范围

共为 36 个主要 Python 文件补充中文注释，重点如下；完整路径见第 6 节：

| 代码组 | 文件 | 新增说明重点 |
| --- | --- | --- |
| 入口 | `train.py` | 配置覆盖、训练/验证选择、resume/tuning、预训练处理 |
| 配置核心 | `workspace.py`、`yaml_utils.py`、`yaml_config.py` | 注册/注入、共享参数、include 顺序、列表替换和惰性创建 |
| 训练控制 | `_solver.py`、`det_solver.py`、`det_engine.py` | 状态管理、AMP/EMA、递归 target、阶段切换与评估 |
| 骨干/融合 | `hgnetv2.py`、`single_hgnetv2_ir.py`、`dual_hgnetv2_add.py`、`dual_hgnetv2_concat.py`、`dual_hgnetv2_lg_dense.py`、`dual_hgnetv2_infranet.py`、`qual_gate.py` | 三尺度、模态顺序、门控公式、条件监督、推理分路与物理解释边界 |
| 数据与评估 | `dataloader.py`、`atrumod_dataset.py`、`dota_dataset.py`、`dota_evaluator.py`、`mmeval.py` | target 组织、配对规则、角度单位、条件缺省、真实 AP 协议 |
| 变换 | `container.py`、`transforms_obb.py`、`transforms_rgbt.py` | 增强阶段、成对锁步、OBB 几何、归一化与 padding 假设 |
| 优化/统计 | `ema.py`、`lr_scheduler.py`、`profiler_utils.py` | 闭包衰减、迭代调度、估算路径与真实效率的区别 |
| 模型主链 | `rtv4.py`、`rtv4_infranet.py`、`hybrid_encoder.py`、`rtdetrv2_obb_decoder.py`、`denoising_obb.py` | 字典接口、双前向、位置编码、旋转采样、查询与 DN |
| 几何/损失/输出 | `utils.py`、`utils_obb.py`、`obb_matcher.py`、`rtv4_obb_criterion.py`、`rtv4_obb_criterion_infranet.py`、`postprocessor.py` | 坐标度量、KLD/Hausdorff、uni-set、条件 loss、辅助权重与还原 |

保留原有 docstring、函数签名和可执行语句；新增注释不会自动实现其中提出的修复或创新。

### 11.2 本次完成的验证与未完成的验证

**已完成的验证范围：**

- 全部 97 个 Python 文件的 AST 语法解析。
- 对比当前源码与 Git HEAD：AST 一致；排除注释/空白行并统一物理行末换行表示后，可执行 token 一致。部分原本无末尾换行的文件新增了末尾换行，不改变执行语义。
- 全部 14 个 YAML 的安全读取和 include 合并检查，核对主要有效参数与缺失组件。
- requirement 语法检查，确认三个无空白 `#` 条目无效。
- 实际 OBB 注意力类和位置编码函数的独立小张量 CPU 验证，复现第 8.1/8.2 节的问题。
- Python 文件清单与 README 逐文件说明的覆盖核对、文档结构检查。

**没有完成、不能冒充已完成的工作：**

- 完整依赖环境安装与兼容矩阵验证。
- 加载真实 ATR/Drone 全部数据、完整训练、完整 AP 重评。
- 全模型运行全部 `tools` 检查脚本、GPU 性能测量和多机运行。
- 上述代码风险修复、新模块实现或研究建议的实验验证。
- 系统外部论文查新和对原论文的忠实复现确认。

本次未改 YAML、依赖清单、权重、`.pyc`、`LICENSE`、用户已有的 `.gitignore` 和 `experiment.md`，未提交 Git commit，也未创建分支。

### 11.3 你可以进一步执行的检查

先安装正确依赖，再运行现有脚本：

```powershell
python tools/test_qual_gate.py
python tools/test_qual.py
python tools/test_all.py
```

这些脚本会创建模型/合成张量并可能占用较多 GPU 内存。通过只代表对应张量/梯度路径工作，不证明 AP 或质量语义正确。

只想检查语法、不生成项目 `.pyc`，可运行：

```powershell
python -B -c "import ast; from pathlib import Path; source_paths = list(Path('.').rglob('*.py')); [ast.parse(source_path.read_text(encoding='utf-8-sig'), filename=str(source_path)) for source_path in source_paths]; print('parsed', len(source_paths), 'files')"
```

不要用会重写已跟踪缓存的 `compileall` 代替本次“只加注释”的完整性检查。语法通过也不能代替 import、数据集和端到端测试。

### 11.4 正式实验应保存的最小信息

- Git 版本/工作区差异、最终合并后的配置与完整命令。
- Python、PyTorch、torchvision、CUDA、GPU 和关键依赖版本。
- 数据划分 ID、类别顺序、坏样本处理、条件缺失率与预训练来源/校验信息。
- 每 seed 的完整训练/验证日志、最佳选择规则、checkpoint 和最终独立测试结果。
- AP 的 IoU 阈值/插值协议、分组样本数、均值/方差及失败样例。
- 实际推理模态、参数统计口径、速度计时方法和峰值显存。

<a id="license"></a>
## 12. 许可证与说明边界

仓库根 `LICENSE` 为 MIT。多个源文件保留上游项目的版权/来源说明，使用与再发布时应保留现有通知并单独确认第三方代码、模型权重和数据集的许可；根许可证不自动替代数据集或外部权重的使用条款。

本 README 是当前代码快照的工程说明与研究审查，不是论文结论、官方复现报告或性能保证。核心判断是：**这个仓库可以作为创新实验底座，但应先建立正确、可复现的几何与评估基线，再通过理论假设、控制实验和文献对照证明具体创新。**
