**不能说你的模型“没有真正的 attention，也没有 token 交互”。更准确的结论是：**

1. **编码器中有标准的图像 token 自注意力。**
2. **解码器每一层都有标准的 object-query 自注意力。**
3. **解码器的可变形跨注意力没有直接计算 Query 与 sampled Key 的 `QKᵀ`，这一点你的观察是对的。**
4. **RGB 与 IR 的融合模块只是相加，没有显式的跨模态 token 注意力。**

因此，需要区分：**整个检测网络的注意力、可变形跨注意力的具体计算方式、RGB–IR 融合的交互方式。**

我核对的是 `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_atrumod_add_baseline.yml:16`，并解析了它的继承配置和构造函数默认值。以下以你当前的实际代码为准，而不是直接套用原版 RT-DETR 的论文架构。

## 一、先把注意力所在的位置列清楚

| 位置 | 当前实际运算 | 是否有标准 `QKᵀ` 注意力 | 交互对象 |
|---|---|---:|---|
| 两套 HGNetv2-B0 | 普通卷积、深度卷积、特征拼接与聚合 | 否 | 卷积感受野内的特征 |
| RGB–IR Add 融合 | 同尺度逐元素相加 | 否 | 同位置、同通道的两模态特征 |
| HybridEncoder 的 S5 | `nn.MultiheadAttention` | **是** | S5 的全部空间 token |
| FPN/PAN | 上下采样、拼接、CSP/卷积融合 | 否 | 不同尺度特征 |
| Decoder 的 self-attention | `nn.MultiheadAttention` | **是** | 300 个检测 query，训练时还包含 DN query |
| Decoder 的 cross-attention | 预测采样位置和权重，聚合多尺度 sampled value | **不直接计算 `QKᵀ`** | 检测 query 与稀疏图像特征 |
| 分类、框回归头 | Linear/MLP | 否 | 每个 query 的类别与框预测 |

**按当前配置构建，模型包含：4 个标准多头自注意力模块，以及 3 个可变形跨注意力模块。** 每个模块都是 8 个头。不是一张“只有卷积和可变形采样”的网络。

---

## 二、整个网络的实际结构

常规输入是：

```text
[B, 6, 512, 640]
```

其中前三通道是 IR，后三通道是 RGB。

按配置推导的完整路径如下，图中的 query 数量不包含训练专用的 DN query：

```text
IR  [B,3,512,640]                  RGB [B,3,512,640]
         │                                 │
   HGNetv2-B0                        HGNetv2-B0
   独立参数                          独立参数
         │                                 │
   IR S3/S4/S5                       RGB S3/S4/S5
         └──────── 同尺度相加 ────────────────┘
                           │
           S3：[B, 256, 64,80]
           S4：[B, 512, 32,40]
           S5：[B,1024, 16,20]
                           │
                    HybridEncoder
                           │
            三尺度分别投影到 224 通道
                           │
             S5 展平为 320 个空间 token
                           │
             1 层标准多头自注意力 + FFN
                           │
                  自顶向下 FPN
                           │
                  自底向上 PAN
                           │
           P3：[B,224,64,80]
           P4：[B,224,32,40]
           P5：[B,224,16,20]
                           │
              Decoder 输入投影与展平
                           │
                [B,6720,224] memory
                           │
               候选分类/框预测 → top-300
                           │
             query：[B,300,224]
             参考框：[B,300,5]
                           │
                  3 层 OBB Decoder
             ┌─────────────────────────┐
             │ query 标准自注意力      │
             │ 旋转可变形跨注意力      │
             │ FFN                     │
             │ 类别预测、参考框更新    │
             └─────────────────────────┘
                           │
             类别：[B,300,11]
             框：  [B,300,5]
                           │
                  PostProcessorOBB
                           │
               类别、分数、像素旋转框
```

总容器的调用顺序可以直接在 `engine/rtv4/rtv4.py:32` 看到：

```text
backbone → encoder → decoder
```

## 三、骨干和 Add 融合：这里确实没有显式跨模态注意力

### 3.1 两套 HGNetv2 完全独立提取特征

IR 与 RGB 分别经过一套 HGNetv2-B0，并不是共享同一套参数。

每条分支的主要阶段为：

| 阶段 | 输出通道 | 输出空间尺寸 | 相对输入步长 |
|---|---:|---:|---:|
| Stem | 16 | 128×160 | 4 |
| Stage1 / S2 | 64 | 128×160 | 4 |
| Stage2 / S3 | 256 | 64×80 | 8 |
| Stage3 / S4 | 512 | 32×40 | 16 |
| Stage4 / S5 | 1024 | 16×20 | 32 |

你的 `return_idx=[1,2,3]` 返回后三个阶段，也就是检测使用的 S3、S4、S5。

HG block 内部主要执行：

```text
连续卷积产生中间特征
        ↓
原输入与中间特征沿通道拼接
        ↓
1×1 卷积聚合、调整通道
        ↓
部分 block 使用残差连接
```

高层使用的 LightConv 包括点卷积与深度卷积，但这些仍是普通卷积，不是 DCN。

依据：`engine/backbone/hgnetv2.py:201`、`engine/backbone/hgnetv2.py:347`。

### 3.2 需要纠正一个 ESE 的细节

**前面 README 对 HGNetv2/ESE 的概括，在这个具体配置上需要收紧。**

虽然 `hgnetv2.py` 定义了 `EseModule`，但是：

- `HG_Stage` 的默认参数是 `agg='se'`。
- `HGNetv2` 创建 stage 时没有覆盖这个参数。
- 当前 `agg='se'` 分支实际是两层 `1×1 ConvBNAct`。
- **没有执行另一分支中的 `EseModule`。**

因此，不能因为文件里存在 ESE 类，就说你的 Add baseline 骨干正在运行 ESE 通道注意力。

依据：`engine/backbone/hgnetv2.py:243`、`engine/backbone/hgnetv2.py:291`、`engine/backbone/hgnetv2.py:465`。

另外，`use_lab=True` 启用的是可学习仿射变换：

\[
y=\mathrm{scale}\cdot x+\mathrm{bias}
\]

它也不是 attention，更不是 Lab 色彩空间转换。实现见 `engine/backbone/hgnetv2.py:27`。

### 3.3 两模态什么时候融合？

实际代码是：

```python
feats_ir = self.backbone_ir(x_ir)
feats_rgb = self.backbone_rgb(x_rgb)

fused_features = [
    f_ir + f_rgb
    for f_ir, f_rgb in zip(feats_ir, feats_rgb)
]
```

见 `engine/backbone/dual_hgnetv2_add.py:64`。

有两个重要结论：

**第一，融合是两套骨干提取结束后，对三个输出尺度分别相加。**

不是：

```text
S3 融合 → 把融合结果送回两条骨干 → 再提取 S4
```

两条骨干在提取过程中没有互相读取对方特征。

**第二，融合公式就是：**

\[
F_s(p,c)=I_s(p,c)+R_s(p,c)
\]

其中位置 \(p\)、通道 \(c\) 都一一对应。

这里没有：

- IR query 对 RGB key 的匹配；
- RGB query 对 IR key 的匹配；
- 跨位置的模态对应搜索；
- 显式的动态模态权重；
- 可学习的跨模态几何配准。

**所以，如果你说“这个融合模块没有真正的跨模态 token 注意力”，这个判断成立。**

但相加仍然融合了信息，而且两套可训练骨干可以调整各自的特征表达，不能进一步推导成“整个网络没有互动”。

---

## 四、HybridEncoder：这里有真正的空间 token 自注意力

### 4.1 为什么你的主 YAML 看起来没有写 attention？

因为有些参数从基础文件继承：

```yaml
use_encoder_idx: [2]
num_encoder_layers: 1
nhead: 8
dim_feedforward: 1024
dropout: 0.
enc_act: gelu
```

来自 `configs/base/rtv2_r50vd.yml:35`。

与主 YAML 合并后，实际是：

```text
hidden_dim          = 224
use_encoder_idx     = [2]
num_encoder_layers  = 1
nhead               = 8
```

`[2]` 指三尺度列表的第三张特征图，也就是 S5。

### 4.2 图像 token 是怎么产生的？

三个尺度先经过 `1×1 Conv + BN`：

```text
S3：256  → 224
S4：512  → 224
S5：1024 → 224
```

然后 S5 被展平：

```text
[B,224,16,20]
       ↓
[B,320,224]
```

这里：

- **320 是 token 数量**；
- **224 是每个 token 的特征维度**；
- 每个 token 对应一个 CNN 特征图空间位置，不要求它一定来自 ViT 的原始 patch embedding。

实现见 `engine/rtv4/hybrid_encoder.py:360`、`engine/rtv4/hybrid_encoder.py:449`。

### 4.3 它确实执行了标准 `QKᵀ`

代码明确创建：

```python
self.self_attn = nn.MultiheadAttention(
    d_model, nhead, dropout, batch_first=True
)
```

前向明确调用：

```python
q = k = self.with_pos_embed(src, pos_embed)
src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
```

见 `engine/rtv4/hybrid_encoder.py:262`、`engine/rtv4/hybrid_encoder.py:278`。

标准缩放点积注意力的核心是：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(\frac{QK^{T}}{\sqrt{d_h}}\right)V
\]

即通过 query–key 内容相关性决定 value 的聚合权重。

在你的模型中：

```text
8 个头
每头维度：224 / 8 = 28
每头空间关系矩阵：320 × 320
```

**S5 上任意一个位置都可以根据内容读取其他 S5 位置。这里是真正的全局空间 token 交互。**

注意两个容易误读的地方：

- `q = k` 表示传入相同的特征，不代表内部 Q/K 投影权重相同；标准多头注意力仍会形成不同的投影。
- 返回的注意力权重被写成 `_`，只是没有拿去显示或额外使用，**不表示注意力没有执行**。

### 4.4 但不是所有尺度都做了全局自注意力

你没有直接对：

```text
S3：5120 个 token
S4：1280 个 token
S5： 320 个 token
```

全部执行各自的全局自注意力，也没有在这里把它们拼成 6720 个 token 做全局自注意力。

实际是：

```text
只有 S5 做标准空间自注意力
        ↓
FPN/PAN 把这些信息传播到其他尺度
```

FPN 自顶向下，PAN 自底向上，使用上采样、下采样、拼接和卷积/CSP 融合。它们有信息混合，但不是标准 `QKᵀ` attention。

当前 `expansion=0.5`，CSP 内部隐藏通道为 112；`depth_mult=0.34` 使相关 block 数取整为 1。实现见 `engine/rtv4/hybrid_encoder.py:200`、`engine/rtv4/hybrid_encoder.py:390`。

**因此，S3/S4 没有直接全局自注意力，不等于它们得不到全局上下文。**

---

## 五、OBB Decoder：既有标准 token 交互，也有你质疑的采样式交互

### 5.1 先形成 6720 个图像 memory token

三尺度在 decoder 内还会经过一次输入投影，再展平、拼接：

\[
64\times80+32\times40+16\times20=6720
\]

得到：

```text
memory：[B,6720,224]
```

见 `engine/rtv4/rtdetrv2_obb_decoder.py:497`。

注意：**这个展平、拼接操作本身不是 attention。**

### 5.2 300 个 query 从哪里来？

模型先对 memory 做候选分类和框预测，再按每个位置的最大类别 logit 选 top-300。

当前 `query_select_method='default'` 的排序依据是分类分数，不是显式计算出来的候选框 IoU。

你的：

```yaml
learn_query_content: False
```

意味着初始 query 内容使用选中 memory 的：

```python
content = enc_topk_memory.detach()
```

见 `engine/rtv4/rtdetrv2_obb_decoder.py:575`。

**这不表示 query 不参与学习，也不表示 attention 被关闭。** 它只是改变 query 初始化来源及初始内容的梯度路径。

每个 query 还带有一个五维参考框：

```text
cx、cy、w、h、归一化角度
```

参考框通过 MLP 生成 query 位置嵌入。

### 5.3 每层先做标准 query self-attention

代码是：

```python
q = k = self.with_pos_embed(target, query_pos_embed)
target2, _ = self.self_attn(
    q, k, value=target, attn_mask=attn_mask
)
```

见 `engine/rtv4/rtdetrv2_obb_decoder.py:240`。

这里的 token 是 **object query**，而不是空间网格 token。

验证时：

```text
300 个 object query
每头关系矩阵：300 × 300
8 个头
3 层 decoder 各执行一次
```

所以它确实让不同检测假设互相读取信息。

例如，一个车辆候选可以结合周围其他候选的语义和位置关系更新自己。不过，**有 query self-attention 不等于严格保证不会产生重复检测。**

### 5.4 接着做旋转可变形 cross-attention

这部分与你关注的问题最相关。

代码为每个 query 预测：

```python
sampling_offsets = self.sampling_offsets(query)
attention_weights = self.attention_weights(query)
```

随后对权重做 softmax。实现见 `engine/rtv4/rtdetrv2_obb_decoder.py:135`。

当前参数是：

```text
3 个尺度
每尺度每头 4 个采样点
8 个头
```

因此：

- 每个 query、每个头有 **12 个采样位置**；
- 每个 query 总计有 **96 个 head-specific 采样位置**；
- 每个头在这 12 个点上共同归一化权重。

采样位置由 query 偏移、参考框中心、宽高和旋转角共同决定，然后使用 `grid_sample` 双线性插值读取图像特征。

最后按权重求和：

\[
y_j =
W_O\operatorname{Concat}_h
\left[
\sum_{\ell,k}
a_{jh\ell k}
V_{h,\ell}(s_{jh\ell k})
\right]
\]

底层采样、加权与求和见 `engine/rtv4/utils.py:171`、`engine/rtv4/utils.py:203`。

Deformable DETR 的核心思路正是围绕参考位置读取少量采样点，避免对全部空间位置建立稠密注意力关系。

### 5.5 你的观察在哪一点上准确？

**当前这一层没有这样的流程：**

```text
采样位置 → 取出 sampled Key
                   ↓
          Query 与 sampled Key 点积
                   ↓
            softmax 得到权重
```

而是：

```text
query → Linear → 采样偏移
query → Linear → 聚合权重

memory → Value projection → 采样 value
                              ↓
                         按权重聚合
```

因此，它的权重不是：

\[
a_{jk}=\operatorname{softmax}(q_j^{T}k_k)
\]

而是由 query 直接预测的。

**如果你把“真正的 attention”严格限定为显式 Q–K 内容匹配，那么这一个 cross-attention 模块确实不满足你的定义。** 原论文公开讨论中也有人提出过类似的 Query/Key 命名质疑。

但不能因此说它完全没有 query–image 交互：

- query 决定在哪里取图像特征；
- 实际 sampled value 决定 query 的更新结果；
- 更新后的 query 在下一层又决定新的采样与权重；
- query 初始内容本身就来自图像 memory。

**它缺的是本次聚合时的直接 sampled-key 匹配，不是完全与图像内容无关。**

### 5.6 它与可变形卷积是什么关系？

你说“借用了可变形卷积思想”有道理：两者都使用可学习偏移读取非固定位置。DCN 的典型做法就是给卷积采样网格增加偏移。

但当前代码运行的不是 `DeformConv2d`：

| 运算 | 采样/匹配方式 | 主要权重形式 |
|---|---|---|
| 普通卷积 | 固定局部网格 | 共享卷积核 |
| 可变形卷积 | 网格加可学习偏移 | 共享卷积核；调制版本还可有 mask |
| 当前可变形跨注意力 | object query 控制多尺度采样 | query 预测的归一化聚合权重 |
| 标准点积注意力 | Query 与 Key 内容匹配 | `softmax(QKᵀ/√d)` |

**概念接近，但计算对象、组织方式与权重机制不完全相同。**

### 5.7 FFN、框更新与输出

每层跨注意力之后还有：

```text
224 → 1024 → 224
ReLU
残差 + LayerNorm
```

FFN 本身是逐 token 的，不建立新的 token 间关系。

每层再预测五维框增量，概括为：

\[
b^{(t+1)}
=
\sigma\left(
\Delta b^{(t)}+\operatorname{logit}(b^{(t)})
\right)
\]

新参考框用于下一层采样，因此是三次迭代定位，而不是一次采样后立即结束。见 `engine/rtv4/rtdetrv2_obb_decoder.py:283`。

最终输出：

```text
pred_logits：[B,300,11]
pred_boxes： [B,300,5]
```

框的角度分量仍是归一化角度，后处理乘 π 转为弧度。

训练时还加入 DN 查询、前两层辅助预测、encoder 候选监督。`num_denoising=100` 是预算参数，实际 DN 数量不是固定恰好 100；普通 query 有 mask，不能直接读取 GT 派生的 DN query。正常验证没有这些 GT 查询。

当前主要检测损失为 focal、五维 L1、KLD，权重分别为 1、5、5。它们属于训练监督，不是 attention 运算。

后处理从 `query×class` 分数中取 top-300，当前没有默认 NMS。见 `engine/rtv4/postprocessor.py:106`。

---

## 六、最关键的问题：有没有 RGB–IR 的“真正 token 交互”？

**没有显式、保留模态身份的跨模态 token attention。**

你可能期待的是：

\[
\operatorname{CrossAttn}_{IR\leftarrow RGB}
=
\operatorname{softmax}
\left(
\frac{Q_{IR}K_{RGB}^{T}}{\sqrt d}
\right)V_{RGB}
\]

也就是：

```text
IR 特征提出问题
        ↓
与 RGB 不同位置的内容比较
        ↓
选择对应的 RGB 信息
        ↓
更新 IR 表征
```

你的当前模型不是这样。

它是：

```text
IR 特征 + RGB 特征
        ↓
模态身份不再显式分开
        ↓
对混合特征做空间注意力
        ↓
query 读取混合后的多尺度 memory
```

所以要区分两个结论：

- **混合后的空间 token 有交互：有。**
- **IR token 显式与 RGB token 建立对应关系：没有。**

而且，后面的非线性和注意力仍然可以形成间接跨模态耦合。简化地忽略位置编码等操作，如果 \(F=I+R\)，那么：

\[
(FW_Q)(FW_K)^T
=
(IW_Q+RW_Q)(IW_K+RW_K)^T
\]

展开后包含 IR–RGB 的交叉项。

但这种混合后的耦合，不等价于分别计算、可追踪的 `IR→RGB` 对应关系，也不保证能从相加特征中重新分离两种模态。

**因此，“融合太简单、缺少显式跨模态对应”是合理批评；“整个网络没有 token 交互”则不准确。**

## 七、最后给你一个严格的架构判断

你的模型可以概括为：

> **双流 CNN 提取特征 + 三尺度直接相加 + 单尺度全局 Transformer 编码 + 卷积金字塔融合 + query 自注意力与旋转稀疏采样解码。**

它不是：

> 只有可变形卷积、所有 token 都独立计算。

但它也不是：

> RGB 与 IR 在多个尺度上进行显式 token 内容匹配的双流 Transformer。

另外，两处已有几何问题仍然值得注意：

- S5 位置编码展平顺序与特征 token 展平顺序不一致：`engine/rtv4/hybrid_encoder.py:428`。
- 512×640 矩形输入下，旋转采样直接旋转归一化偏移，缺少宽高度量修正：`engine/rtv4/rtdetrv2_obb_decoder.py:173`。

**这些影响注意力“按什么位置进行交互”，但不代表注意力没有执行。**

如果你后续以此做创新，准确的切入点应该是 **“增加显式跨模态内容对应、质量选择或 sampled-key 匹配”**，而不是“给一个没有 attention 的模型首次加入 attention”。

本次仅分析配置和源码，没有修改项目文件，也没有运行完整训练。