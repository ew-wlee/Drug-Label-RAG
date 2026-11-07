# MSBD 5018 · Group 18 · Domain-Aware FDA Drug Label QA

**领域感知的 FDA 药品标签问答 / Domain-aware question answering over FDA drug labels**

本项目比较 No RAG、General RAG 和 Domain-Adapted RAG，研究标签章节切分、混合检索与重排能否改善证据覆盖和最终回答。FDARxBench 用于短答、检索和拒答评估。RxQA 用于四选一准确率评估，使用独立的 openFDA 语料。

This project compares No RAG, General RAG, and Domain-Adapted RAG to study how section-aware chunking, hybrid retrieval, and reranking affect evidence coverage and answer quality. FDARxBench evaluates short answers, evidence retrieval, and abstention. RxQA evaluates four-option accuracy against a separate openFDA corpus.

## 1. 实验设计 / Experimental design

| 系统 / System | 检索与生成 / Retrieval and generation |
|---|---|
| No RAG | 直接回答题目。RxQA 同时给 A–D 选项。 / Answer directly from the question. Include options A–D for RxQA. |
| General RAG | 固定窗口 → MiniLM 余弦检索 → 共享重排器。 / Fixed windows → MiniLM cosine retrieval → shared reranker. |
| Domain-Adapted RAG | 按章节切分 → MiniLM + BM25 → RRF → 共享重排器。 / Section-aware chunks → MiniLM + BM25 → RRF → shared reranker. |

- **中文**：窗口上限 160 个空白分隔词，重叠 24 词。两组都在检索文本中加上药品名。检索只传题干，RRF 使用两路各前 20 名及常数 60。主实验使用冻结的 MS MARCO MiniLM 对各组前 10 个候选重排，生成最多看到 5 块证据。各条件使用同一个生成模型和相同解码参数。
- **English** — Windows contain up to 160 whitespace-delimited words with 24 words of overlap. Both indexes prefix the drug name. Retrieval uses the question stem. RRF combines the top 20 from each ranking with a constant of 60. A frozen MS MARCO MiniLM cross-encoder reranks up to 10 candidates per system, and the generator receives up to five excerpts. Each paired comparison holds the generator and decoding settings fixed.

FDARxBench 使用官方标签及段落坐标。openFDA 管线保留标签版本和章节，按标签 ID 及同一标签集和版本内的文本摘要去重，并使用 SQLite FTS5 和磁盘映射的精确余弦索引。训练、重排微调和近似索引是后续扩展。

FDARxBench uses the official labels and passage coordinates. The openFDA pipeline retains label versions and section names, deduplicates by label ID and by text hash within a label set and version, and uses SQLite FTS5 plus disk-mapped exact cosine search. Training, reranker fine-tuning, and approximate indexes are future extensions.

## 2. 项目导航 / Repository map

| 路径 / Path | 用途 / Purpose |
|---|---|
| [experiment.py](experiment.py) | FDARxBench 切分、检索、重排与证据评分 / FDARxBench chunking, retrieval, reranking, and evidence scoring |
| [answer.py](answer.py) | 本地短答生成、自动评分及续跑 / Local short-answer generation, grading, and resumption |
| [compare.py](compare.py) | 按题配对 bootstrap / Paired question-level bootstrap |
| [openfda_rxqa.py](openfda_rxqa.py) | openFDA 导入、索引和 RxQA 选择题评估 / openFDA ingestion, indexing, and RxQA evaluation |
| [data/openfda/manifest.json](data/openfda/manifest.json) | 14 卷快照、大小和 SHA-256 / Snapshot metadata, sizes, and SHA-256 for 14 archives |
| [results](results) | 指标、逐题预测和检索证据 / Metrics, per-question predictions, and retrieved evidence |
| [review](review) | 人工盲评材料与独立揭盲表 / Human review packet and separate answer key |
| [OPENFDA_PLAN.md](OPENFDA_PLAN.md) | 详细设计和扩展计划 / Detailed design and extension plan |
| [select_subset.py](select_subset.py) | 覆盖型子集目录、题目映射、盲审与固定背景 / Coverage-subset selection |
| [Step 1 配置](configs/step1_main.json) · [准入校验](scripts/validate_step1.py) | 冻结主实验并自动验收 / Freeze and validate Step 1 |
| [报告草稿 / Report draft](report/Group18_Report_Draft.pdf) · [展示稿 / Slides](presentation/Group18_Presentation.pptx) | FDARxBench 阶段交付 / FDARxBench deliverables |

### Step 1 主运行 / Step 1 main run

当前运行为 `runs/step1-coverage-20261106-r2/`。由于官方 2026-11-06 分卷在下载时发生同日刷新，实际快照为 262,909 条；原始 `data/openfda/manifest.json` 保留不变，实际哈希和两条新增记录差异由 `source_manifest.json` 记录。

本运行使用统一冻结的 10,000 份覆盖型标签：6,379 份核心相关标签、2,000 份困难负例和 1,621 份固定背景。Fixed/Section 索引分别为 395,316/454,013 chunks。300 道 RxQA 均保留，296 道为 `stem_exact`、4 道为 `option_only`；900 行 prompts 不含 gold，答案独立保存在 `rxqa_answer_key.json`。该结果只能解释为固定覆盖型子集上的配对比较，而非全量 openFDA corpus-wide retrieval。

```sh
python select_subset.py prepare-sources
python select_subset.py catalog
python select_subset.py map
# 按 rxqa_blind_review.json 完成不接触答案表的审核
python select_subset.py select
python openfda_rxqa.py index --zip data/openfda/drug-label-*.json.zip \
  --selected-labels runs/step1-coverage-20261106-r2/subset_manifest.json \
  --out runs/step1-coverage-20261106-r2/index --device mps \
  --encode-batch-size 32 --max-chunks 900000 --max-index-bytes 12884901888
python openfda_rxqa.py prepare --index runs/step1-coverage-20261106-r2/index \
  --csv rxqa/rxqa_openfda.csv --out runs/step1-coverage-20261106-r2/rxqa_prompts.jsonl \
  --device mps --block-size 8192
python scripts/validate_step1.py
```

Step 1 不调用 Ollama 或生成模型。资源数字仅是本地工程测量，不是研究变量。

## 3. 环境准备 / Setup

使用 Python 3.12、sentence-transformers、Transformers、NumPy、ijson 和 SQLite FTS5。短答和选择题生成使用 Ollama。Apple Silicon 可使用 MPS 加速。Qwen3.5 2B、4B 和 9B 已在 M4、24 GB Mac 上完成短答实测，建议串行加载。以下命令从本项目目录运行。

Use Python 3.12, sentence-transformers, Transformers, NumPy, ijson, and SQLite FTS5. Ollama serves the generators for both tasks. Apple Silicon can accelerate indexing and reranking with MPS. Qwen3.5 2B, 4B, and 9B have been tested for short-answer generation on an M4 Mac with 24 GB of unified memory. Load models sequentially. Run commands from this project directory.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c 'import sqlite3; sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE probe USING fts5(text)")'
ollama pull qwen3.5:2b
ollama pull qwen3.5:4b
# Optional for the shared model-size subset
ollama pull qwen3.5:9b
```

如 Ollama 未启动，在另一终端运行 `ollama serve`。脚本默认访问 `http://127.0.0.1:11434`，自定义端口时传 `--url`。首次运行检索会下载模型权重。

If Ollama is stopped, run `ollama serve` in a separate terminal. Scripts default to `http://127.0.0.1:11434`. Pass `--url` for a custom port. Retrieval model weights download on the first run.

## 4. FDARxBench 复现 / Reproducing FDARxBench

固定版本含 17,207 道题和 699 条可索引标签。检索样本为 90 题，事实、多跳、拒答各 30 道，种子为 5018。检索指标使用其中 60 道可回答题。

The pinned revision contains 17,207 questions and 699 indexable labels. The retrieval sample contains 90 questions, with 30 factual, 30 multihop, and 30 refusal questions at seed 5018. Retrieval metrics use the 60 answerable questions.

```sh
git clone https://github.com/xiongbetty/FDARxBench.git
git -C FDARxBench checkout 019785d2802c4ceec583e4d4b82b902d7a8f9910

python experiment.py --data FDARxBench/data --out runs/dense
python experiment.py --data FDARxBench/data --out runs/rerank \
  --from-retrieval runs/dense --reranker cross-encoder/ms-marco-MiniLM-L6-v2

for model in 2b 4b 9b; do
  per_task=10
  if [ "$model" = 9b ]; then per_task=5; fi
  python answer.py runs/rerank/retrieval.jsonl "runs/qwen-${model}/predictions.jsonl" \
    --model "qwen3.5:${model}" --systems no_rag general_rerank domain_rerank \
    --per-task "$per_task"
done

python answer.py runs/qwen-2b/predictions.jsonl runs/qwen-2b/judged.jsonl \
  --judge --model qwen3.5:4b --systems no_rag general_rerank domain_rerank
```

补充消融分别测试 BM25 切分、共享 Qwen 重排器和按题配对的差值。非 Apple GPU 环境可省略 Qwen 重排命令的设备和半精度参数。

The following commands reproduce the lexical chunking ablation, the shared Qwen reranker sensitivity check, and paired differences. For non-Apple GPU environments, omit the Qwen reranker device and half-precision flags.

```sh
python experiment.py --data FDARxBench/data --out runs/lexical --lexical-only
python experiment.py --data FDARxBench/data --out runs/qwen-rerank-shared \
  --from-retrieval runs/dense --per-task 5 --reranker Qwen/Qwen3-Reranker-0.6B \
  --reranker-device mps --reranker-fp16
python compare.py runs/lexical/retrieval.jsonl fixed_bm25 section_bm25
python compare.py runs/rerank/retrieval.jsonl general_rerank domain_rerank
python compare.py runs/qwen-2b/predictions.jsonl general_rerank domain_rerank --answer-f1
```

## 5. openFDA × RxQA 复现 / Reproducing openFDA × RxQA

### 数据与容量 / Data and capacity

- **中文**：全量数据来自 2026-11-06 的同一 openFDA 快照，14 个 ZIP 合计约 1.86 GB，解压后的 JSON 合计约 8.75 GB。脚本直接从 ZIP 流式解析，无须另存 JSON。试跑使用独立的 `pilot-2026-11-05.zip`。若压缩包未随项目提供，请从官方来源获取并核对清单。动态下载地址可能返回新快照。
- **English** — The complete corpus comes from a single openFDA snapshot dated 2026-11-06. Its 14 ZIPs total about 1.86 GB, and the expanded JSON totals about 8.75 GB. The indexer streams directly from ZIP. The pilot uses the separate `pilot-2026-11-05.zip`. If archives are absent, obtain them from the official source and verify against the manifest. Live URLs can return a newer snapshot. [openFDA downloads](https://open.fda.gov/apis/drug/label/download/)

### 试跑 / Pilot

```sh
git clone https://github.com/Google-Health/rxqa.git
git -C rxqa checkout 203979bf928c1fe9e45aa89e2ed76c3d2a63b503

python openfda_rxqa.py index --zip data/openfda/pilot-2026-11-05.zip \
  --out runs/rxqa-pilot --limit 1000 --device mps
python openfda_rxqa.py prepare --index runs/rxqa-pilot --csv rxqa/rxqa_openfda.csv \
  --out runs/rxqa-pilot/prompts.jsonl --per-difficulty 3 --device mps
for model in 2b 4b; do
  python openfda_rxqa.py evaluate --input runs/rxqa-pilot/prompts.jsonl \
    --out "runs/rxqa-pilot/qwen-${model}.jsonl" --model "qwen3.5:${model}"
done
```

无 MPS 时省略 `--device mps`。索引目录必须是新目录。换模型时复用同一份 prompts，并指定新的预测文件。RxQA 从 `correct_idx` 读取零起始选项下标，并校验对应选项文本。无法解析的模型输出计错且单列。

Omit `--device mps` when unavailable. The index command requires a new output directory. Reuse the same prompts across models and write predictions to separate files. RxQA uses the zero-based `correct_idx` and verifies that it matches the answer text. Unparseable model outputs count as incorrect and are reported separately.

### 主运行范围 / Main-run scope

本项目不构建全量 openFDA 向量索引；正式主运行使用上述冻结的覆盖型子集。pilot 的随机 1,000 标签结果仅用于流程验证，不与 10,000 标签主运行混报。主运行索引实测约 3.98 GB，300 题 prepare 的峰值 RSS 约 1.63 GB；这些是本地复现测量，不属于研究条件。

The main run uses the frozen coverage subset above rather than a full openFDA vector index. The random 1,000-label pilot remains a pipeline check and is not mixed with the main run.

## 6. 指标与结果 / Metrics and results

| 指标 / Metric | 定义 / Definition |
|---|---|
| Recall@k | 前 k 块与官方证据词范围有重叠时计命中，按题计算命中的 gold 段落比例。 / Per-question fraction of gold passages with any word-range overlap in the top k. |
| MRR@10 | 第一次命中的排名倒数，无命中记零。 / Reciprocal rank of the first hit in the top 10, or zero. |
| Evidence80@k | 至少覆盖 80% 证据词位置的 gold 段落比例，可跨块合并。 / Fraction of gold passages with at least 80% word-position coverage across retrieved chunks. |
| Short-answer EM / F1 | 归一化后的完全匹配和词重叠 F1。拒答单独统计。 / Normalized exact match and token-overlap F1. Abstentions are reported separately. |
| RxQA accuracy | 正确选项数除以题数，同时按难度分层。 / Correct choices divided by questions, also broken down by difficulty. |

FDARxBench 拒答题不计入证据检索指标。自动盲评基于统一的参考答案和 gold evidence。实际引文是否支持具体主张由人工核对。配对区间用按任务分层的 5,000 次 bootstrap，描述本次样本。

FDARxBench refusal questions are excluded from retrieval metrics. Blind model grading uses the same reference answers and gold evidence across systems. Human reviewers check whether the actual cited excerpts support each claim. Paired intervals use 5,000 task-stratified bootstrap replicates and describe the evaluated sample.

| 已发布实验 / Published run | No RAG | General RAG | Domain RAG |
|---|---:|---:|---:|
| FDARxBench Evidence80@5, 60 answerable questions | — | 52.5% | 64.3% |
| FDARxBench Recall@5, 60 answerable questions | — | 64.9% | 66.0% |
| FDARxBench 2B token F1, 20 answerable questions | 0.267 | 0.324 | 0.366 |
| FDARxBench 4B token F1, 20 answerable questions | 0.167 | 0.394 | 0.421 |
| RxQA pilot 2B accuracy, 12 questions | 6/12 | 5/12 | 5/12 |
| RxQA pilot 4B accuracy, 12 questions | 5/12 | 6/12 | 4/12 |

FDARxBench 共保存 225 条生成结果。2B 和 4B 的主表用 30 道题，9B 用三款模型共享的 15 题子集。两个主表的 Domain 对 General F1 差值区间均包含零。RxQA 的 72 条预测来自抽样语料，属于流程验证。公开 RxQA CSV 缺少逐题 gold passage，因此本项目对 RxQA 汇报答案准确率。

FDARxBench includes 225 saved generations. The 2B and 4B main runs use 30 questions. The 9B run uses the 15-question subset shared across all three model sizes. The paired F1 intervals for Domain versus General include zero in both main runs. The 72 RxQA predictions are a pipeline pilot on sampled labels. The public RxQA CSV lacks per-question gold passages, so the project reports answer accuracy for RxQA.

[检索汇总 / Retrieval metrics](results/retrieval.json) · [短答与共享子集 / Short answers and shared subset](results/answers) · [RxQA 逐题预测 / RxQA per-question predictions](results/rxqa_pilot) · [人工复核 / Human review](review/README.md)

## 7. 常见问题 / Troubleshooting

- **本地连接失败 / Connection refused** — 确认 Ollama 已启动，模型已下载，端口与 `--url` 一致。 / Confirm Ollama is running, the model is pulled, and the port matches `--url`.
- **分卷混用 / Mixed snapshots** — 全量命令仅匹配 `drug-label-*.json.zip`。将旧版 pilot ZIP 单独用于试跑。 / Use only `drug-label-*.json.zip` for the full run and keep the older pilot ZIP separate.
- **恢复运行 / Resuming** — 生成和评分可用相同输入、模型和输出文件续跑。更换设置时使用新输出文件。未完成的索引请用新目录重建。 / Resume generation and grading with the same input, model, and output. Choose a new output for changed settings and rebuild an incomplete index in a fresh directory.
- **复编报告 / Rebuilding the report** — 在安装 LaTeX 后运行 `cd report && latexmk -pdf main.tex`。 / With LaTeX installed, run `cd report && latexmk -pdf main.tex`.

## 8. 数据与模型来源 / Data and model sources

- [FDARxBench repository](https://github.com/xiongbetty/FDARxBench) · [Paper](https://arxiv.org/abs/2603.19539)
- [RxQA repository](https://github.com/google-health/rxqa) · [openFDA Drug Labeling](https://open.fda.gov/apis/drug/label/)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [MS MARCO MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) · [Qwen3 Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [Qwen3.5 on Ollama](https://ollama.com/library/qwen3.5)
