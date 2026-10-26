# ComfyUI-MiniMaxH3-Speed-Linux

MiniMax H3 视频+音频联合生成的推理加速节点，可在 Linux 上运行，是开源 Python 实现。

## 由来

[TE-Speed-MiniMaxH3](https://github.com/caoddgao/TE-Speed-MiniMaxH3)（作者 B站 TETAE）只发布了 Windows 闭源二进制
`nodes.pyd`，Linux 上无法加载。本项目**没有**使用或反编译其中的代码，只沿用了它在界面上可见的节点 ID、参数名和默认值，
所以原来含有 `TE-Speed-MiniMaxH3` 节点的工作流可以直接打开。加速算法是基于 ComfyUI 原生 H3 模型代码独立实现的，
效果和速度不会与 TE 原版完全一致。与 TE 作者没有关联。

## 原理

每一步都完整运行第一个和最后一个 transformer block。根据第一个 block 残差相对上一步的变化量来判断：变化小时，
中间的 block（1 … N-2）不运行，改用上一次完整计算时缓存的中间残差。cond 和 uncond 各自独立缓存。

- 采样开始时才安装 block patch，并放在最外层，所以与官方 `Block Sparse Attention`、H3 ControlNet 等 block patch
  连接的先后顺序没有影响：完整步照常调用它们，跳过的步连同它们一起跳过（它们的效果已包含在缓存残差里）。
- 支持 ComfyUI block prefetch / Comfy Compiler（缓存张量分配在 malloc graph 之外）。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `processing_control_value` | 0.08 | 累计变化阈值。越大跳得越多、越快，细节损失越大 |
| `processing_percent_1` | 0.1 | 从采样进度的这个位置开始允许跳过 |
| `processing_percent_2` | 0.9 | 到这个位置后不再跳过 |
| `mcs` | 2 | 最多连续跳过的步数 |
| `device` | auto | 中间残差缓存位置：`auto` 显存够就放显存，否则放锁页内存；`gpu`；`cpu` |
| `mode` | standard | `4-step LoRA` / `8-step LoRA`：总步数 ≤10 时使用短轨迹策略（只在第 1 至倒数第 2 步之间跳、最多连跳 1 步、变化量按步长归一化）；11 步及以上同 standard。两个 LoRA 选项行为相同 |

缓存大小约为 `序列长度 × 5376 × 2 字节`，每个 cond/uncond 分支一份。长视频显存不够时选 `auto` 或 `cpu`。

## 用法

`Load Diffusion Model → (LoRA / Block Sparse Attention …) → TE-Speed-MiniMaxH3 → 采样器`。
日志会输出 `TE-Speed-MiniMaxH3: skipped middle blocks on X/Y model passes`。用 `--verbose DEBUG` 可以看每一步的判断。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/caoddgao/ComfyUI-MiniMaxH3-Speed-Linux.git
```

重启 ComfyUI。不需要额外的依赖。需要支持 MiniMax H3 的新版 ComfyUI。

## 许可证

Apache-2.0
