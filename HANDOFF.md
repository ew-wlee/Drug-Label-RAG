# 接手指南 · Group 18

**当前状态**：Step 1 已冻结，Step 2 的两款模型、三种条件、300 题共 1,800 条正式预测已完成。详见 [最终报告](report/Group18_Report_Final.pdf) 、[答辩 PPT](presentation/Group18_Presentation_Final.pptx) 与 [正式结果](runs/step2-rxqa-20261107/)。项目架构、结果和复跑命令见 [README](README.md)。

- GitHub 仓库：https://github.com/ew-wlee/Drug-Label-RAG
- OneDrive 大文件：https://hkustconnect-my.sharepoint.com/:f:/r/personal/zlioe_connect_ust_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fzlioe%5Fconnect%5Fust%5Fhk%2FDocuments%2FMSBD5018%2DGroupProject&share=cgrEwqUfjGtVRbXLvs3HMmwtEgUBztFNxtQ4dbos%2DffhxWKkmA

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
git clone https://github.com/ew-wlee/Drug-Label-RAG.git
cd Drug-Label-RAG

# 2. 放大文件（OneDrive 下载后拖回对应目录）
#    runs/                 → 覆盖/合并到 runs/
#    drug-label-*.zip      → 放到 data/openfda/
# 对照 runs/step1-coverage-20261106-r2/source_manifest.json 中实际冻结分卷的 SHA-256
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

1. **别覆盖** `runs/step1-coverage-20261106-r2/` 里已有的文件。新的预测放进单独的 Step 2 运行目录。
2. **不要把答案喂给模型**。`rxqa_prompts.jsonl` 不含 gold，答案单独在 `rxqa_answer_key.json`，只用于打分。
3. **改样本 / 提示词 / 参数 → 新建 run ID**，不要在冻结目录里改。索引命令的 `--out` 必须是新目录。
4. **不重跑** `prepare-sources` / `catalog` / `map` / `select` / 索引，也不构建全量 openFDA 向量索引。

---

## 5. 正式结果与剩余交接

- 用 `python scripts/summarize_rxqa.py runs/step1-coverage-20261106-r2 runs/step2-rxqa-20261107` 复核 1,800 条预测、分难度准确率并重算配对区间。
- 下载包缺失的 `labels.sqlite` 已在本地下载目录重建。[恢复记录](runs/step2-rxqa-20261107/index-recovery.json) 列出原文件与新文件的 SHA-256。重建索引重新通过 [Step 1 九项验收](runs/step2-rxqa-20261107/recovered-index-validation.json)。正式答题可直接使用冻结提示。
- 两位组员可按 [盲评说明](review/README.md) 独立评分。交卷后打开系统对应表，再汇总分歧。报告目前将人工评分列为待完成。
- 如课程要求补交答辩材料或上传结果，由组员确认提交版本和渠道。

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
