# 在双 DGX Spark 上运行 MiniMax H3

[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Platform: 2x DGX Spark](https://img.shields.io/badge/platform-2x%20DGX%20Spark-76B900)](#measured-topology)
[![Architecture: ARM64](https://img.shields.io/badge/architecture-ARM64-informational)](#measured-topology)
[![Interconnect: RoCEv2](https://img.shields.io/badge/interconnect-RoCEv2-76B900)](#how-one-video-spans-both-machines)

一台 MiniMax H3 FL2VA 视频，由两台 NVIDIA DGX Spark 协同生成。

本仓库是[单 Spark 方案](https://github.com/joeynyc/MiniMax-H3-DGX-Spark)的一个独立实验性扩展。它在固定的 vLLM-Omni 镜像之上补充了原镜像没有的多机扩散执行器，Ray 把两个 rank 分别放在两台 Spark 上，NCCL 通过专用 RoCEv2 链路传输张量集体通信。

> [!IMPORTANT]
> MiniMax H3 **并不适用**本仓库的 Apache-2.0 许可证。其社区许可目前排除美国、欧盟、英国和韩国，并限制在适用领土之外使用和展示输出。在下载、运行或展示模型输出之前，请阅读 [MODEL-LICENSE.md](MODEL-LICENSE.md) 并从 MiniMax 获得所需授权。本仓库不包含模型权重，也不包含已生成的媒体文件。

## 实测拓扑

| 角色 | 主机 | Fabric 地址 | 分布式 rank |
|---|---|---:|---:|
| Spark 1 / head/API | `spark-head` 示例 | 私有 fabric 地址 | 0 |
| Spark 2 / peer | `spark-peer` 示例 | 私有 fabric 地址 | 1 |

DiT 使用双向 Ulysses 序列并行。两张 GPU 共同参与同一个去噪轨迹；rank 0 额外持有编码器、VAE 和最终 API 响应。

## 本地部署（2026-08-19，本 fork）

本 fork 实际部署在工作区根目录 `/home/xujie/workspace/dgx-spark-minimax-h3` 的双 Spark 集群上。角色**已翻转**：本地工作机（`spark-0d97`）只跑轻量的 rank 1，API 主机是另一台 Spark：

| 角色 | 主机 | Fabric 地址 | 分布式 rank | 加载后内存占用 |
|---|---|---:|---:|---:|
| API / rank 0（编码器、VAE、输出） | `xujie2` / `spark-ac8f` | 10.100.65.1 | 0 | ~80 GiB |
| rank 1（DiT 一半） | local / `spark-0d97` | 10.100.65.2 | 1 | ~60 GiB |

- API 地址：`http://10.100.65.1:8000/v1`；Ray 面板：`http://10.100.65.1:8265`（仅私有 fabric，默认无认证）。
- Fabric：`enP2p1s0f0np0`，HCA `roceP2p1s0f0`，两端 RoCEv2 GID index 均为 3；NCCL all_gather 实测约 21 GB/s。
- 镜像：base 镜像 `minimax-h3-dgx-spark:sm121-fp8` 从 companion commit `8bd7628` 本地重制为 `sha256:e498adce…`（上游验收用的 image ID 无法跨机器复现，因为 Docker image ID 会嵌入构建元数据；但上游 digest、层血缘和运行时版本仍会 fail-closed 校验）。服务镜像 `minimax-h3-2x-dgx-spark:experimental` 当前为 `sha256:02d75101…`，双机一致。
- 在此集群构建需要 host 网络和一个可访问的 PyPI 镜像：

  ```bash
  H3_BUILD_NETWORK=host \
  H3_BUILD_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
  H3_BUILD_PIP_TRUSTED_HOST=mirrors.aliyun.com \
  SYNC_WORKER=1 ./scripts/build-image.sh
  ```

- 实测验收（固定 768×448 T2VA，20 步）：首请求 70.652 s / 73.551 s（惰性区域编译），热请求 47.914 s；角色翻转前后输出字节稳定（SHA-256 `212a43c8…`）。
- 双节点模型文件逐字节一致（81/81 文件 SHA-256 匹配，约 135 GiB）。本机已安装 `ffmpeg`/`ffprobe`，用于 `make smoke` / `make verify`。
- 操作说明已打包为 agent skill `dgx-spark-2x-h3-api`（启动、使用、重建、排障），见 `~/.kimi-code/skills/dgx-spark-2x-h3-api`。

### Ref2VA 多参考输入（2026-08-22）

派生镜像通过覆盖上游 H3 pipeline（`patches/minimax_h3_pipeline.py`）并给 OpenAI video API 打补丁（`patches/allow-mixed-ref-inputs.patch`），让 Ref2VA 分区支持完整的 Comfy 风格参考集：最多 **9 张图片**、**3 个视频**、**3 个独立音频**，任意组合（仅 `task=ref2va`；fl2va/t2va 仍保持上游单参考规则）。

- multipart `input_references` 按内容类型/文件名嗅探（`h3_multinode.ref_inputs`）：图片以 PIL 对象跨 Ray 传输，视频持久化在 API 节点（只有 rank 0 读取），音频在 API 进程解码为 `(waveform, sample_rate)` 元组，无需共享主机路径。
- `audio_reference`（`file://` JSON，通过 `patches/allow-file-audio-url.patch`）仍然可用，并与上传的音频合并；`file://` 路径必须同时存在于两台 Spark 上。
- 超出上限的请求在 GPU 工作前即返回 HTTP 400。
- 条件标签顺序与 Comfy 一致：先图片，再每个视频（先声音标签再视频标签），最后独立音频。
- 参考图对齐遵循 ComfyUI 的 `ref_image_size`（通过 `extra_params` 传入）：`match`（默认）按生成画布面积匹配；`max` 保留最长 2048 px 短边，以更高 token 代价换取更强身份保真。两种模式都保持宽高比且只缩小不放大，与 ComfyUI 完全一致。

  `match` 是对输出画布的**非均匀**缩放：参考图宽高比必须与 `width`/`height` 一致，否则主体会被拉伸/压扁。1:1 的参考图搭配方形画布（例如 768×768，或 992×992，32 对齐且仍在模型原生约 1.03MP 画布上限内）不会产生宽高比畸变；`max` 则无论画布如何都保持参考图宽高比。

## 本 fork 相对上游的优化

上游固定方案以在线 FP8 + 惰性区域编译跑 FL2VA T2VA。本 fork 已将在线服务迁移到 **Ref2VA + 常驻 turbo LoRA + INT8 权重**，并修复了若干实测热点。以下数字均为本双 Spark 集群（SM121、eager、CUDNN_ATTN）的单实验室测量，不是厂商声明。

### 服务形态

- **双 rank 目录化 LoRA 服务**：`ref2v` turbo 适配器（rank 384、208 个模块）从每主机目录在启动时静态加载；`allowed_steps` 为 {4, 6, 8}，默认 4 步（`flow_shift=12`、`audio_flow_shift=3`）。
- **INT8 ConvRot DiT 权重**：`patches/comfy_kitchen_int8.py` 实现 comfy-kitchen `LinearMethodBase`，加载 Comfy-Org 预量化 checkpoint（int8 qdata + 每行 fp32 scale）。DiT 权重内存减半；请求峰值内存从 BF16 + `max` 参考的约 99.5 GiB 降到约 77 GiB，速度与 BF16 在 SM121 上基本持平。默认使用 W8A16（反量化 + 普通 GEMM，与 ComfyUI 加 LoRA 行为一致）；`H3_INT8_W8A8=1` 可启用融合 W8A8 核（已在 3 图 `max` 工作流上验证：W8A8 77.0 s，W8A16 82.6 s，质量相当）。
- **双 Spark 上关闭 compile**：区域编译未排除 Ulysses SP2 的 all-to-all；第一次编译请求会在 mid-step 创建新的 NCCL 通信器，导致 rank 不匹配、inline executor 崩溃（2026-08-24 实测）。本 fork 在编译能正确排除 all-to-all 之前使用 `eager`。

### 延迟优化

- **文本编码器 TP2**：Qwen3-VL 编码器切到双 rank（`H3_TEXT_ENCODER_TP_SIZE=2`），rank-0 峰值从 101.5 GiB 降到约 74 GiB，双机负载均接近约 80 GiB。
- **异步 MP4 编码**：视频编码放到 worker 线程；编码期间 `/health` 保持在 15 ms 以内（之前事件循环在 swap 压力下会阻塞数十秒）。
- **VAE decode stack tiling**（`H3_VAE_STACK_TILING=1`）：每个 rank 的 256 px 空间 tile 按每个时间 chunk 合并成一次 decoder forward。10 s / 832×480 解码从 36.1 s 降到 **21.9 s**（请求总时长 162.8→148.7 s）；2 s 解码从 5.7 s 降到 3.8 s；帧已目视验证一致。将 decoder tile size 提高到 512 被拒绝：更快（15.8 s），但出现全画面网格伪影（decoder 按 256 px tile 训练）。
- **worker 侧 uint8 导出**：帧在 `decode()` 内部于 GPU 上转 uint8（与旧 float 链路同一条 op 链，已验证逐字节一致），不再以 fp32 形态跨 Ray pickle/plasma——10 s / 832×480 负载缩小 4 倍（1.16 GiB → 291 MiB），结果尾部（forward 结束 → API 返回）从 1–8.4 s（内存压力下方差很大）降到 **约 1.5 s**。
- **分阶段可观测性**：每个响应都带 `X-Stage-Durations` 头（`encode_prompt` / `diffuse` / `decode`），并可选开启 torch.profiler 钩子覆盖 `diffuse()` 和 `decode()`（`H3_TORCH_PROFILER_DIR` 或 `/tmp/h3_profiler_on` 哨兵）。当前 10 s / 832×480 / 3 参考拆解：encode_prompt 2.7 s、diffuse 111.9 s、decode 22.0 s、尾部约 1.5 s、MP4 编码 1.3 s。

### 冷启动

- **mmap→匿名内存中转再上传 CUDA**：safetensors 零拷贝 mmap view 的 pageable H2D 拷贝在 GB10 上只有约 150–200 MiB/s（单核，与页缓存冷热无关），导致 32 GiB INT8 DiT shard 加载耗时 222.84 s。`patches/minimax_h3_transformer.py` 现在先把每个 CPU view `clone()` 到匿名内存（约 15 GB/s），再上传（约 10.8 GB/s），合成约 6.9 GB/s：DiT 权重加载从 222.84 s 降到 **16.76 s**，launch→ready 从 **约 10 分钟降到约 4.3 分钟**。

### ComfyUI 输出一致性

- **参考 VAE 编码**使用 VAE 后验均值 FP32，而非采样后的 FP16 latent（`patches/ref-encode-posterior-mean.patch`）。
- **Qwen vision tower 以 FP32 计算**（`patches/vision-tower-fp32.patch`）；bf16 舍入会明显模糊建筑等细纹理，人脸影响较小。

### 当前实测数字（INT8 W8A8、eager、镜像 `02d75101`）

| 工作负载 | 总耗时 | 峰值内存 |
|---|---:|---:|
| 768×448 / 2 s / 1 图 + 音频 / `match` / 4 步 | 25.7 s | 75.9 GiB |
| 832×480 / 2 s / 3 图 + 音频 / `max` / 4 步 | 77.0 s | 77.0 GiB |
| 832×480 / 10 s / 3 图 + 音频 / `match` / 4 步 | 142.0 s | ~80 GiB |

## 已验证结果

最终公开发布的验收在 2026 年 8 月 4 日完成，双节点使用同一派生镜像：`sha256:09e6521356bbbb635048228d30e78a36c65352a48f7620c921d5aeff2d21b90b`。每次冷启动后的首请求包含惰性区域编译；第二请求为热测量。

| 配置 | 冷启动就绪 | 编译预热 | 热请求 |
|---|---:|---:|---:|
| cuDNN + 区域编译，全算力 | 588.98 s | 70.337 s | 46.574 s |
| cuDNN + 区域编译，均衡 Cache-DiT | 584.91 s | 55.412 s | 30.578 s |

四次固定输入 T2VA 请求均生成 56 帧、768×448、24 fps 的 H.264/AAC MP4。完整 FFmpeg 解码、非静音音频、中点目视检查全部通过。两张 GPU 在生成期间均活跃，Ray 报告两个健康节点，三个容器零重启、无 OOM。API 和 Ray 面板仅监听配置的私有 head 地址。

更早的匹配测试建立了更广泛的性能与质量结果：

- 首次分布式验收分别用时 68.783 s 和 64.888 s，而可比的单 Spark 请求为 154.956 s——客户端侧约快 2.3 倍。
- cuDNN attention + 区域编译热请求平均 48.689 s，比 64.226 s 的 Torch-SDPA/eager 基线快 24.2%。
- 1344×768、50 步全算力负载从 2,281.532 s 降到 1,353.506 s。输出完整解码，与此前同种子结果的视频 SSIM 为 0.9134。
- 均衡 Cache-DiT 配置完成同一负载仅 608.991 s——比全算力延迟低 55.0%，生成速率高 2.22 倍。同种子输出与全算力相比视频 SSIM 0.888、平均视频 PSNR 27.04 dB，音频基本无变化。

这些是来自一个双机实验室的测量结果，不是厂商基准或上游支持保证。Cache-DiT 是经目视检查的速度/质量权衡，不是无损模式。完整验收记录见 [docs/RESULTS.md](docs/RESULTS.md)。

## 一个视频如何跨两台机器

```text
client -> Spark 1 API -> Ray executor
                         |-- rank 0 / Spark 1 / cuda:0 --\
                         |                               | Ulysses SP + NCCL/RoCE
                         `-- rank 1 / Spark 2 / cuda:0 --/
                                      |
                              rank 0 编码 MP4 -> client
```

这是模型并行，不是两个独立任务：每个去噪步都切分到两个 rank 上。

## 前置条件

- 两台 ARM64 DGX Spark，FL2VA checkpoint 在两台节点上位于相同绝对路径。
- head 和 peer 的免密 SSH 别名。
- 每台节点专用的 fabric 地址、网卡、RoCE HCA 和当前生效的 GID index。重启后不要假设两端 GID index 相同。
- 支持 GPU 和 `/dev/infiniband` 的 Docker。
- 来自单 Spark 项目的已知良好 base 镜像 `minimax-h3-dgx-spark:sm121-fp8`。
- 在你所在领土使用 MiniMax H3 的授权。

checkpoint 和 base 镜像不在本仓库分发。路径、主机、地址、端口、网卡、HCA 和 GID 可在 `.env` 中配置；示例使用不可路由的文档地址，必须替换。

## 快速开始

```bash
git clone git@github.com:Ajie16/MiniMax-H3-2x-DGX-Spark.git
cd MiniMax-H3-2x-DGX-Spark
cp .env.example .env
# 编辑 .env：两台主机、fabric、共享路径、许可证确认。

make audit
make build
./scripts/start-two-sparks.sh
./scripts/wait-ready.sh
make status
```

先按[单 Spark 仓库](https://github.com/joeynyc/MiniMax-H3-DGX-Spark)构建 base 镜像 `minimax-h3-dgx-spark:sm121-fp8`。`make build` 会把派生双节点镜像传送到 peer，并 fail-closed 校验接受的 base 血缘、运行时版本以及双机派生镜像 ID 是否一致。接受的 companion commit、上游 digest、本地 base image ID 和精确运行时包版本记录在[可复现性指南](docs/REPRODUCIBILITY.md)。

最终验收的冷启动就绪时长为 584.91–588.98 秒（FL2VA 时期）；当前 Ref2VA INT8 eager 配置冷启动约 4.3 分钟。`wait-ready.sh` 会等待模型加载完成，然后检查两个 Ray 节点、HTTP 健康、三个容器和精确模型身份。API 仅监听配置好的私有 head 节点地址的 `8000` 端口；同步生成使用 `POST /v1/videos/sync`。

上游默认 fast profile 会惰性编译；**本 fork 使用 `eager`，因为区域编译会破坏双 rank Ulysses 路径**（见上文"本 fork 相对上游的优化"）。eager 下没有编译预热请求，首请求和第二请求速度接近。

默认启动器关闭跨步缓存以获得最大保真度。如需启动实测过的均衡 Cache-DiT 配置：

```bash
./scripts/start-cache-dit-profile.sh
./scripts/wait-ready.sh
```

`./scripts/start-two-sparks.sh` 可切回无缓存配置。两个启动器只替换本实验的三个容器；模型数据和基准产物会保留。

仅在模型许可允许生成和展示的领土内运行：

```bash
make smoke
make verify
```

仅停止本实验：

```bash
make down
```

## 组件

- `h3_multinode/executor.py`：基于 Ray 的跨主机扩散执行器。
- `h3_multinode/worker.py`：每主机单 GPU 的全局 rank 感知扩散 worker。
- `docs/ARCHITECTURE.md`：控制路径、数据路径、失败模型和范围。
- `docs/REPRODUCIBILITY.md`：接受的镜像血缘、commit 和运行时版本。
- `scripts/preflight.sh`：fail-closed 的主机、模型、镜像、端口和 fabric 检查。
- `scripts/start-two-sparks.sh`：启动 Ray rank 基础设施和 API。
- `scripts/start-cache-dit-profile.sh`：启动实测过的可选均衡缓存配置。
- `scripts/stop-two-sparks.sh`：仅停止本实验的容器。
- `scripts/status.sh`：校验两个容器、Ray 节点、健康和模型身份。
- `scripts/wait-ready.sh`：等待冷加载完成，然后运行完整状态检查。
- `scripts/smoke-t2va.sh`：固定单视频验收请求。
- `scripts/benchmark-smoke.sh`：保留固定输入性能产物和元数据。
- `scripts/benchmark-hq-beach.sh`：重复固定 50 步质量工作负载，不覆盖结果。
- `scripts/check-runtime-versions.py`：校验接受镜像的运行时包集合。
- `scripts/verify-output.sh`：流、完整解码、音频和哈希验证。
- `scripts/nccl-smoke.py`：独立的双 rank all-reduce 测试。
- `scripts/public-audit.sh`：跟踪文件、凭据模式、大小和语法检查。

已知良好的单 Spark 部署不会被本仓库修改。

## 实验边界

- 自定义执行器故意固定为恰好两台各一张 GPU 的主机。
- 已验收场景为同步、batch-size-one 的 FL2VA 生成，未覆盖并发生产流量或任意扩散模型。
- API/head rank 内存占用更高，因为编码器和输出路径未均匀分片。
- 两次接受运行中，固定种子解码中点帧完全一致。MP4 字节哈希不同（容器相差 8 字节），因此不声称为逐字节 MP4 确定性。
- 接受的 cuDNN/编译配置在实测热运行中是确定性的，但更换 attention 后端会改变数值轨迹，可能使同种子采样出不同场景。
- Cache-DiT 在选定去噪步复用 block residual。实测热运行中确定，但会故意改变像素；需要严格全算力保真度时，使用默认无缓存启动器。
- 本仓库 patch 了 vLLM-Omni 内部执行器钩子，每次上游镜像变更都应重新验证。
- Ray、其面板、API 和动态 worker 流量使用 host 网络且无认证。面板和 API 仅绑定配置的 head fabric 地址；其余分布式端口仍要求相互信任、已防火墙隔离的节点。详见 [SECURITY.md](SECURITY.md)。

## 可复现性声明

此处的“可复现”指：文档化的拓扑能重复启动，两个 rank 都完成一次生成，输出通过相同的媒体检查。它并不意味着 MP4 字节完全一致、兼容任意硬件，或上游支持多节点 H3 配置。

## 上游与引用

- [MiniMax H3 model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [Companion single-Spark compatibility recipe](https://github.com/joeynyc/MiniMax-H3-DGX-Spark)

仓库代码采用 Apache-2.0。MiniMax H3 权重和输出仍受其单独许可约束。引用信息见 [NOTICE](NOTICE)。
