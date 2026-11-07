# MSBD 5018 · openFDA 双基准实验方案

## 研究问题与交付

在同一个生成模型上，用同一批题配对比较 No RAG、General RAG 和 Domain-Adapted RAG。RxQA 衡量药品选择题准确率，FDARxBench 衡量短答正确性和证据覆盖。主实验用冻结模型。若时间允许，训练领域重排器作为单独消融。已有 FDARxBench 实验保留为复现基线，可复用 `experiment.py`、`answer.py` 和 `compare.py` 中的排名、生成、评分和配对分析逻辑。

## 本机运行范围

Apple M4、24 GB 统一内存。Qwen3.5 2B、4B、9B 和 Qwen3-Reranker-0.6B 均已实测。9B 运行中 Ollama 日志记录的峰值约 15.1 GiB。按模型串行运行，复用检索缓存。优先使用 4B 跑完整实验，2B 开发调试，9B 做固定子集或后续扩跑。Gemma 4 E4B 和 MedGemma 4B IT 使用兼容的量化版本先各试跑同一批题，记录模型版本、量化格式、峰值和耗时。DeepSeek V4/V4.1 如纳入，用远程推理，固定模型 ID，禁用搜索和工具，独立标注后端。当前盘约有 17 GiB 可用，先测第一卷展开体积、每条标签平均 chunk 数以及每万块的索引成本，再安排全量盘位。可用外接 SSD 存数据和索引。

## 技术栈

- Python 3.12、`zipfile`、`hashlib`、`ijson` 用于下载验证和流式解析。
- SQLite 保存标签版本、章节、证据偏移和运行元数据。SQLite FTS5 的 BM25 用于词法检索，先做小样本分词检查。NumPy 用于精确余弦基线，FAISS CPU 的 HNSW 用于全库索引，固定参数并抽样对照精确检索。
- sentence-transformers、PyTorch MPS、all-MiniLM-L6-v2 用于归一化向量编码，MS MARCO MiniLM 或 Qwen3-Reranker-0.6B 用于冻结交叉编码重排。先实测 MPS 和 CPU 的速度。
- Ollama 本地运行生成模型，沿用现有接口。可选兼容 MLX 或 GGUF 模型后端。训练若获准，单独在有 GPU 的环境使用 sentence-transformers 与相应训练依赖。
- JSONL 保存逐题输出，CSV/JSON 保存汇总。pytest 覆盖数据解析、证据偏移、评估和续跑。配置保存 Git commit、数据哈希、模型标识、量化、分词器、随机种子和上下文预算。

## 模块与命令设计

| 模块 | 入口与输出 |
|---|---|
| `data/openfda.py` | `fetch` 接收 manifest 或指定快照 URL，校验分卷数量、哈希和 zip 完整性。解压用流式写入，`inspect --part 1 --limit 3` 只读元数据和少量记录 |
| `data/normalize.py` | `normalize` 从 `results.item` 流式读取，写出 labels、sections 和去重映射，保存处理计数 |
| `retrieval/chunks.py` | `chunk` 用同一份规范化标签生成 fixed 和 section 两种索引输入，保留原始证据坐标 |
| `retrieval/index.py` | `index` 分批写向量和 FTS5，保存索引配置、进度和稳定 chunk ID，重复运行可复用 |
| `retrieval/search.py` | `retrieve` 运行 dense、BM25、RRF、cross-encoder，缓存每题各阶段候选、得分和耗时 |
| `benchmarks/` | 两个适配器统一出题目 ID、题干、选项或参考答案、题型及可用 gold evidence。单独生成证据对齐报告 |
| `generation/` | 复用 `answer.py` 的模型调用。以 dataset、qid、model、system 为运行键，保存 prompt、实际证据和原始回答，可按题恢复 |
| `evaluation/` | MCQ 判分、短答判分、拒答指标、检索覆盖率、盲评、配对 bootstrap，输出各任务的样本数及失败类别 |

模块名是实施目标。先在现有代码里增加适配器和配置，规模增长后再拆目录。

## openFDA 数据与切分

已抽查指定第一卷的前 3 条记录。外层为 `meta` 与 `results`。快照的 `meta.results.total` 显示 262,907 条，第一卷 limit 为 20,000。第一卷压缩包约 130 MiB，ZIP 目录列出的解压体积约 603 MiB。按 14 卷等体积粗估，压缩包和解压 JSON 合计约 10 GiB，实际以各卷大小为准。样本中的 `id`、`set_id`、`version`、`effective_time` 为字符串，`openfda` 内含品牌、通用名、给药途径、NDC 等字段。正文按章节字段存储，多为文本数组。处方药样本有 `warnings_and_cautions`、`adverse_reactions` 和多种 `*_table`，OTC 样本有 `warnings`、`purpose`、`do_not_use`。导入时允许字段缺失。分卷数量随快照核对。

1. 先保留所有版本和原始引用。按 `set_id + version` 识别标签版本，在同一版本内对规范化文本做精确去重并保留别名映射。常规检索视图可选每个 set_id 的最新版本，评估需要的旧版本保留可索引。不同剂型、强度或给药途径的标签分别保留。
2. 提取章节、产品标识、版本和偏移。清洗空白和标记，保留数字、单位、列表与表头。表格随对应章节进入索引，拆表时携带表头。
3. General 固定窗口。将规范化的章节按稳定顺序串接后切窗，可跨章节。Domain 先按字段分章节，再按段落切长章节，保持相同 token 上限和重叠配置。两组候选都包含同等药品标识，生成都使用同一 top-k 与总 token 预算。
4. 384 维 float32 向量每百万条约 1.54 GB，只计原始向量。另算原始 zip、解压 JSON、文本库、第二套向量索引和 HNSW 图。先在第一卷记录数、向量条数与每阶段字节数，再估算全库峰值。

## 对照设计

- No RAG 使用题干。RxQA 同时给 A–D 选项。
- General RAG 使用固定窗口、MiniLM dense 检索、共享冻结重排器。
- Domain-Adapted RAG 使用章节窗口、同一 dense encoder 与 BM25，经 RRF 合并，使用相同重排器及候选预算。选项是否参与检索在两组统一，默认只用题干，生成时给全选项。
- 消融依次比较固定 dense、章节 dense、章节 hybrid 和加共享 reranker。微调另列，使用药物分组隔离 train/dev/test。只有完成基线且有足够的正负样本与 GPU 预算时，才做交叉编码器训练。困难负例从训练药物的其他章节或近似药品检索获得，人工抽查可能的假负例。

## 两套评估

**RxQA** 使用公开的 300 道 OpenFDA 四选一题。加载 CSV 的 A–D、`correct_idx` 和难度字段，模型输出单一选项，无法解析的输出计错并单列。主指标是 accuracy 和按难度分层的 accuracy，比较用按题配对区间。公开 CSV 缺少逐题 passage gold，检索指标仅用于额外人工标注的子集。[RxQA 数据与 notebook](https://github.com/google-health/rxqa)

**FDARxBench** 读 QA JSONL 的 `qid`、`task`、`answer`、`set_id`、`context`。使用官方生成和 judge 脚本作参考，另保留项目的同题闭卷拒答对照并标注口径。主检索评估用官方标签快照，证据按 passage ID 和原始词范围对应，计算 Recall@k、MRR 和 Evidence80。将 openFDA 扩库结果单独报告。若要求同样在 openFDA 上评估短答，先用 set_id、版本和文本跨度映射 gold，报告覆盖率及版本冲突，只有已对齐题计算证据指标。另做 oracle evidence 运行，可区分检索失误与给足证据仍答错。拒答标签源于指定参考文档，扩库后若找到新证据需单独复核。[FDARxBench 数据与评估脚本](https://github.com/xiongbetty/FDARxBench)

短答汇报 EM/F1、固定盲评 rubric 的正确且完整率、证据支持率、拒答率和可回答题的过度拒答率。人工评分员检查实际引文、剂型、单位、限定条件及未获支持的主张。检索证据是否充分和生成是否忠实分别评估。所有模型和系统使用预先固定的共同题集，模型规模不同的实验子集单独标注。

## 执行顺序

第一卷抽样与小索引 → 核对全量磁盘预算和快照 → 流式导入 14 个分卷 → RxQA 和 FDARxBench 的少量同题冒烟 → 冻结测试样本与参数 → Qwen 2B/4B 主矩阵 → 9B 与 Gemma 家族共同子集 → 消融、盲评、错误分析。全量数据和训练可选项的验收以实际资源预算为准。
