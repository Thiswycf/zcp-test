# GPU 选择

`zcp-test` 在 Conda 环境中常驻设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。查看物理卡：

```bash
conda activate zcp-test
zcp-test gpu list
```

输出同时包含 `index`（`nvidia-smi` 序号）、`pci_order`、PCI Bus ID、UUID、型号、空闲显存和利用率。运行 manifest 以 UUID 和 Bus ID 标识物理设备，不用容易被 `CUDA_VISIBLE_DEVICES` 重映射的逻辑序号作为实验身份。

## 选择方法

```bash
# 默认：空闲显存降序、利用率升序、PCI Bus ID 升序
zcp-test evaluate ... --gpu auto

# 限制型号
zcp-test evaluate ... --gpu auto --gpu-model "RTX 4090"

# 使用 nvidia-smi index、UUID 或 PCI Bus ID
zcp-test evaluate ... --gpu 4
zcp-test evaluate ... --gpu GPU-1f28b951-...
zcp-test evaluate ... --gpu 00000000:98:00.0

# 要求至少 20 GiB 空闲显存
zcp-test train ... --min-free-memory 20480

# CPU 或保留旧逻辑设备语义
zcp-test evaluate ... --device cpu
zcp-test evaluate ... --device cuda:0
```

自动选择后，程序将物理 UUID 写入 `CUDA_VISIBLE_DEVICES`，因此进程内设备始终是 `cuda:0`。GPU 锁位于 `~/.cache/zcp-test/gpu-locks/`；`--gpu auto` 遇到锁冲突时会按相同排序规则尝试下一张满足型号和显存条件的卡，全部候选均被占用才会失败，且绝不会悄悄切到 CPU。显式指定 INDEX、UUID 或 PCI Bus ID 时不会改选其他卡。

该锁只协调同一用户下遵循 `zcp-test` 锁协议的进程，不是系统级独占锁，无法阻止其他用户或普通 CUDA 程序使用同一张卡。`CUDA_VISIBLE_DEVICES` 绑定按 CLI 短进程设计，嵌入式调用不应在同一 Python 进程中连续改绑多张 GPU。

现有环境可用下列命令检查常驻变量：

```bash
conda env config vars list -n zcp-test
```
