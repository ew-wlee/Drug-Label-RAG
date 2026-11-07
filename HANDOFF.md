# 接手指南 · Group 18

**一句话**：Step 1 已做完并冻结，现在轮到 Step 2 —— 用同一份 900 行题目，跑 2B / 4B 两个模型共 1800 条预测，然后算准确率。

> 先把下面两个空填上，再发给队友：
> - GitHub 仓库：https://github.com/ew-wlee/Drug-Label-RAG
> - OneDrive 大文件：https://hkustconnect-my.sharepoint.com/:f:/r/personal/zlioe_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzlioe%5Fconnect%5Fust%5Fhk%2FDocuments%2FMSBD5018%2DGroupProject&share=cgrEwqUfjGtVRbXLvs3HMmwtEgUBztFNxtQ4dbos%2DffhxWKkmA

---

## 1. 先拿到三样东西

| 要拿什么 | 从哪拿 | 放到哪 |
| --- | --- | --- |
| 代码、配置、冻结小产物（约 30 MB） | `git clone` GitHub 仓库 | 得到 `drug-label-rag/` 目录 |
| `runs/`（4.0 GB）、`data/openfda/` 的 14 个 zip（1.7 GB） | OneDrive | 按原目录名放回去 |
| Python 3.12 + Ollama | 自己装 | 见第 2 步 |

**不用别人传的**：`.venv/`（1.1 GB）和 `.cache/`（371 MB）各人自己生成，传了也用不了。

---

## 2. 五步跑起来

```sh
# 1. 拿代码
git clone [填仓库地址]
cd drug-label-rag

# 2. 放大文件（OneDrive 下载后拖回对应目录）
#    runs/                 → 覆盖/合并到 runs/
#    drug-label-*.zip      → 放到 data/openfda/
# 校验一下没下错版本（对照 data/openfda/manifest.json 的 sha256）
shasum -a 256 data/openfda/drug-label-0001-of-0014.json.zip

# 3. 装环境
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v     # 5 个测试应全过

# 4. 验收（必须输出 passed）
python scripts/validate_step1.py

# 5. 跑 Step 2（另开一个终端先启动 Ollama）
ollama serve
```

---

## 3. 目录都是干什么的

| 目录 / 文件 | 作用 |
| --- | --- |
| `openfda_rxqa.py` | openFDA 建索引、生成题目、跑模型评估 |
| `select_subset.py` | 冻结 10,000 份标签子集（`prepare-sources` / `catalog` / `map` / `select`） |
| `experiment.py` `answer.py` `compare.py` | FDARxBench 阶段的检索、短答、配对比较 |
| `configs/step1_main.json` | 冻结的主实验配置，别改 |
| `runs/step1-coverage-20261106-r2/` | **冻结的 Step 1 成果**：900 行 prompts、答案键、各种清单、3.7 GB 索引 |
| `data/openfda/` | 14 卷原始快照 zip + `manifest.json`（含 sha256） |
| `rxqa/rxqa_openfda.csv` | 300 道 RxQA 题 |
| `results/` `report/` `review/` | 已有结果、报告 LaTeX 源码、人工盲评材料 |
| `README.md` | 完整复现命令（第 3 节环境、第 4/5 节复现、第 7 节排错） |

`FDARxBench/` 是第三方仓库，不随 GitHub 分发；需要时按 `README.md` 第 4 节 `git clone` 后 `checkout 019785d...`，并保留其 Apache-2.0 `LICENSE`。

---

## 4. 四条红线

1. **别覆盖** `runs/step1-coverage-20261106-r2/` 里已有的文件。只按第 5 节新增 `qwen-*.jsonl` 预测文件。
2. **不要把答案喂给模型**。`rxqa_prompts.jsonl` 不含 gold，答案单独在 `rxqa_answer_key.json`，只用于打分。
3. **改样本 / 提示词 / 参数 → 新建 run ID**，不要在冻结目录里改。索引命令的 `--out` 必须是新目录。
4. **不重跑** `prepare-sources` / `catalog` / `map` / `select` / 索引，也不构建全量 openFDA 向量索引。

---

## 5. 下一步：跑 Step 2

确认运行时和 Qwen3.5 2B/4B 的版本、量化、digest 并记进运行清单，**不要改 prompts**：

```sh
python openfda_rxqa.py evaluate \
  --input runs/step1-coverage-20261106-r2/rxqa_prompts.jsonl \
  --out runs/step1-coverage-20261106-r2/qwen-2b.jsonl \
  --model qwen3.5:2b \
  --answer-key runs/step1-coverage-20261106-r2/rxqa_answer_key.json

python openfda_rxqa.py evaluate \
  --input runs/step1-coverage-20261106-r2/rxqa_prompts.jsonl \
  --out runs/step1-coverage-20261106-r2/qwen-4b.jsonl \
  --model qwen3.5:4b \
  --answer-key runs/step1-coverage-20261106-r2/rxqa_answer_key.json
```

要交的东西：overall / 分难度 accuracy、invalid rate、按题配对区间。RxQA 没有逐题 gold passage，**不要报检索 Recall**。主矩阵做完再谈扩展题、人工盲评和消融。

---

## 6. 结果怎么交回来

| 内容 | 方式 |
| --- | --- |
| 代码、文档、配置改动 | 开分支 → PR，别直接推 `main` |
| 预测 jsonl、新索引等大文件 | 放 OneDrive，PR 里写路径 + sha256 |
| 新实验 | 新 run ID，例 `runs/step2-20261108-<姓名>/` |

---

## 7. 卡住了看哪

- 连接被拒 → 另开终端 `ollama serve`，确认模型已 `ollama pull`。
- 快照版本不对 → 全量命令只匹配 `drug-label-*.json.zip`，旧 pilot zip 单独用。
- 索引建到一半 → 换新目录重建，别在原目录续。
- 细节 → `README.md` 第 7 节，或工作区的 `CODEBUDDY.md`。
