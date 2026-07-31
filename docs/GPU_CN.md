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

# 锁冲突时最多等待 30 秒；0 表示立即失败
zcp-test train ... --gpu-lock-timeout 30

# CPU 或保留旧逻辑设备语义
zcp-test evaluate ... --device cpu
zcp-test evaluate ... --device cuda:0
```

自动选择后，程序将物理 UUID 写入 `CUDA_VISIBLE_DEVICES`，因此进程内设备始终是 `cuda:0`。GPU 锁位于 `~/.cache/zcp-test/gpu-locks/`；`--gpu auto` 遇到锁冲突时会按相同排序规则尝试下一张满足型号和显存条件的卡，全部候选均被占用才会失败，且绝不会悄悄切到 CPU。显式指定 INDEX、UUID 或 PCI Bus ID 时不会改选其他卡。

该锁只协调同一用户下遵循 `zcp-test` 锁协议的进程，不是系统级独占锁，无法阻止其他用户或普通 CUDA 程序使用同一张卡。`CUDA_VISIBLE_DEVICES` 绑定按 CLI 短进程设计，嵌入式调用不应在同一 Python 进程中连续改绑多张 GPU。

不要把 `~/.cache/zcp-test/gpu-locks/*.lock` 文件的存在误判为占用：`flock` 是附着在打开文件描述符
上的内核锁，进程退出后自动释放，锁文件可以继续存在。正常释放会清空文件内的 owner PID。若看到
GPU 空闲但怀疑被锁，应使用 `flock -n <lock-file> -c true` 做无破坏探测；返回成功表示可用，返回失败
才表示某个活跃进程持锁。不得直接 `rm`，因为删除路径不能解除旧 inode 上的活跃锁，反而可能让两个
进程分别锁住不同 inode。高成本 launcher 必须按实际任务/lane 持锁，完成 lane 立即释放，禁止
supervisor 在数据预检或等待其他 lane 时预占四张卡。锁 holder 会在任务子进程一侧关闭继承 FD，
Python 也注册了 fork 后关闭逻辑，防止 pipeline、DataLoader 或孤儿 worker 延长锁生命周期。

`--gpu-lock-timeout` 必须是非负秒数。正数是本次选择过程获取合格锁的总等待时间，不是每张卡
各自重新计时；`--gpu auto` 会在剩余时间内尝试下一候选，显式 index/UUID/Bus ID 不会换卡。
`--device` 是兼容性覆盖，会绕过物理 GPU 选择和锁。

现有环境可用下列命令检查常驻变量：

```bash
conda env config vars list -n zcp-test
```
