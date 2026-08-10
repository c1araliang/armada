---
name: aris
description: Universal ARIS autonomous research workflow router and assistant. Handles all research tasks including paper writing, proof checking, proof orchestrating, logic auditing, claim verification, formula derivation, literature discovery, ablation planning, auto-review loop, and presentation prep. (支持中英文指令 / Supports English & Chinese inputs)
---

# ARIS Master Research Router (Auto-Research-In-Sleep) / ARIS 主科研路由调度器

Welcome to **ARIS** — the unified research coordinator. This master skill routes user requests in either English or Chinese to specialized ARIS sub-skills located in `${CLAUDE_SKILL_DIR}/subskills/` based on user intent.

---

## ⚠️ MANDATORY OUTPUT RULE / 强制输出规范 (CRITICAL)

**At the VERY BEGINNING of your response (the very first line), you MUST output a single-line summary explicitly listing the sub-skill(s) activated:**

- **Chinese / 中文格式**: `📌 **[ARIS 路由调度] 本次已激活并调用的子技能：`<subskill_name1>`, `<subskill_name2>`**`
- **English / 英文格式**: `📌 **[ARIS Master Router] Activated Sub-skill(s): `<subskill_name1>`, `<subskill_name2>`**`

*Example / 示例*: `📌 **[ARIS 路由调度] 本次已激活并调用的子技能：`proof-checker`, `proof-orchestrator`**`

---

## 🎯 Intent Routing Table / 意图路由映射表

When receiving a user prompt in English or Chinese, classify the intent, inspect the corresponding sub-skill file using `view_file` at `${CLAUDE_SKILL_DIR}/subskills/<subskill_name>/SKILL.md`, and execute the pipeline:

| User Intent (EN / CN) | Sub-skill | Path |
|---|---|---|
| **Proof Checking & Symbol Verification / 逻辑推导审查、符号记号与推导树门禁** | `proof-checker` / `proof-orchestrator` | `${CLAUDE_SKILL_DIR}/subskills/proof-checker/SKILL.md` |
| **Formula Derivation & Math Proof / 公式严谨推导、数学证明与 Lemma 撰写** | `formula-derivation` / `proof-writer` | `${CLAUDE_SKILL_DIR}/subskills/formula-derivation/SKILL.md` |
| **Adversarial Logic Review & Flaw Hunting / 审稿人视角对抗逻辑审查、漏洞排除** | `kill-argument` | `${CLAUDE_SKILL_DIR}/subskills/kill-argument/SKILL.md` |
| **Claim Auditing & Precision Drafting / 论文 Claim 逻辑映射、结论拟定与防止过度承诺** | `paper-claim-audit` / `claims-drafting` / `result-to-claim` | `${CLAUDE_SKILL_DIR}/subskills/paper-claim-audit/SKILL.md` |
| **Theoretical Novelty Check / 理论新颖性与逻辑独创性审查** | `novelty-check` | `${CLAUDE_SKILL_DIR}/subskills/novelty-check/SKILL.md` |
| **Academic Integrity & Compliance / 学术规范合规与 46 种逻辑/证据漏洞预审** | `integrity-forensics` | `${CLAUDE_SKILL_DIR}/subskills/integrity-forensics/SKILL.md` |
| **Paper Writing & Drafting / 论文撰写、大纲与草稿生成** | `paper-writing` / `paper-plan` | `${CLAUDE_SKILL_DIR}/subskills/paper-writing/SKILL.md` |
| **Ablation Planning & Experiment Audit / 消融实验设计与实验结果严谨审计** | `ablation-planner` / `experiment-audit` | `${CLAUDE_SKILL_DIR}/subskills/ablation-planner/SKILL.md` |
| **Auto Review Loop & Self-Optimization / 跨模型自动盲审循环与自演化** | `auto-review-loop` / `meta-optimize` | `${CLAUDE_SKILL_DIR}/subskills/auto-review-loop/SKILL.md` |
| **Citation Audit & Ref Verification / 参考文献真实性与引用精准度审计** | `citation-audit` | `${CLAUDE_SKILL_DIR}/subskills/citation-audit/SKILL.md` |
| **Literature Search & Multi-source Retrieval / 广域文献检索与 AlphaXiv/S2 脉络** | `openalex` / `arxiv` / `alphaxiv` / `semantic-scholar` / `research-lit` | `${CLAUDE_SKILL_DIR}/subskills/openalex/SKILL.md` |
| **Idea Generation & Wiki Knowledge Graph / 创新点生成与研究记忆图谱** | `idea-creator` / `research-wiki` / `wiki-enrich` | `${CLAUDE_SKILL_DIR}/subskills/research-wiki/SKILL.md` |
| **Architecture Diagram & Visual Spec / 论文架构图设计与 Mermaid 绘图** | `figure-spec` / `mermaid-diagram` | `${CLAUDE_SKILL_DIR}/subskills/mermaid-diagram/SKILL.md` |
| **Presentation & Oral Talk Prep / 会议 Oral 演讲备忘录与 PPT 极致排版** | `paper-talk` / `slides-polish` | `${CLAUDE_SKILL_DIR}/subskills/paper-talk/SKILL.md` |
| **HTML Report Rendering / 成果单文件网页报告渲染导出** | `render-html` | `${CLAUDE_SKILL_DIR}/subskills/render-html/SKILL.md` |

---

## 🚀 Execution Workflow / 执行步骤

1. **Output Activation Line**: Start your response with the mandatory single-line sub-skill activation notice.
2. **Read Sub-skill Instructions**: Call `view_file` to load `${CLAUDE_SKILL_DIR}/subskills/<subskill>/SKILL.md`.
3. **Execute Protocol & Report**: Execute the specialized research protocol and return structured, high-quality results.
