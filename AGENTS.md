# AGENTS.md — ReviewPulse Engineering and Research Contract

> **Applies to:** human contributors, ChatGPT/Codex, local coding agents, IDE agents, CI automation and any assistant editing this repository. Last source audit: 2026-10-22. Read `README.md`, source code and relevant JSON reports before changing claims, behavior or metrics.

## 0. The non-negotiable mandate

ReviewPulse is a **research-backed Indonesian e-commerce review intelligence prototype**, not a production-validated classifier or a general-purpose psychological inference platform. Preserve scientific integrity, traceability, privacy, and compatibility with the shipped models. Do not invent features, data, confidence scores, citations, deployment links or evaluation improvements.

When an instruction conflicts with observed repository files, **inspect and report the discrepancy**. When a document conflicts with a machine-generated research artifact, reconcile by reviewing the experiment—never quietly substitute an attractive number.

## 1. Source-of-truth hierarchy

1. **Actual code and tests** (`app/`, `src/reviewpulse/`, `tests/`) establish implemented behavior.
2. **Committed research artifacts** (`reports/official_test_metrics.json`, `reports/model_selection.json`, `reports/source_audit.json`, `reports/final_summary.json`) establish recorded metrics and selection.
3. **Original training notebook and its logs** (not committed at inspection time) establish the precise training execution when supplied and verified.
4. **README.md and docs/** communicate validated facts and limitations; correct them after code/evidence changes.
5. **Issue descriptions, prompts, plans and chat histories** are intentions—not evidence.

Before reporting new scores, cite the new experiment identifier, dataset revision, split, denominator, metric calculation, hardware/environment, and exactly which artifact contains the result.

## 2. Current verified snapshot

- Repository: `MENSTRUE/ReviewPulse`, primary branch `main`, MIT code license.
- Dataset: PRDECT-ID, recorded 5,400 review rows, 11 columns, 0 missing source cells, 7 full-row duplicates, 95 duplicate review-text rows, 5,305 unique raw review texts.
- Sentiment labels: Negative 2,821; Positive 2,579.
- Emotion labels: Happy 1,770; Sadness 1,202; Fear 920; Love 809; Anger 699.
- Research split: grouped normalized review text; train 3,782, validation 807, held-out internal test 811.
- Model selection: validation macro-F1; sentiment `TFIDF-LinearSVC`, emotion `TFIDF-LogReg`.
- Reported internal test: sentiment accuracy 0.9506781751, macro-F1 0.9506510892; emotion accuracy 0.6263871763, macro-F1 0.5885637039. Test N=811 each.
- Model artifacts: `models/reviewpulse_sentiment_tfidf.joblib`, `models/reviewpulse_emotion_tfidf.joblib`.
- Application: Streamlit single review + bulk CSV + descriptive visualizations and exports.
- Assets actually committed: 2 dataset distribution plots, 2 evaluation confusion matrices, 1 negative-review term plot.
- Training `.ipynb`, CI workflow, hosted deployment and calibrated predictions have **not** been confirmed in the checked repo.

**Always re-inspect remote `main` before asserting this snapshot remains current.** Do not conflate a local unpushed file with a publicly committed file.

## 3. Before → Now → Next change-report format

Every substantial code/documentation PR or milestone report must state:

```text
BEFORE: What actually existed, what failed, which evidence showed it.
NOW: Exactly which files changed, commands executed, tests performed, and observed outputs.
NEXT: Specific remaining risks, blocked requirements, and acceptance criteria.
```

The NOW section must not call a proposed feature shipped; NEXT items must not be presented as completed; no runtime estimate or promise of future unattended work.

## 4. Repository ownership map

| Location | Contract |
|---|---|
| `app/streamlit_app.py` | Presentation and user flows; call domain functions, avoid retraining. |
| `src/reviewpulse/config.py` | Resolved paths/configuration; avoid hard-coded Windows-only locations. |
| `src/reviewpulse/preprocessing.py` | Minimal input cleanup consistent with fitted pipeline. |
| `src/reviewpulse/inference.py` | Trusted model loading, deterministic batch inference, honest confidence behavior. |
| `src/reviewpulse/analytics.py` | Descriptive aggregates only. |
| `src/reviewpulse/visualization.py` | Figure construction from real prediction data. |
| `tests/` | Unit, integration, regression and contract testing. |
| `models/` | Approved, locally trusted trained sklearn artifacts. Never commit arbitrary third-party pickle. |
| `reports/` | Immutable or versioned experiment evidence. Do not silently overwrite scores. |
| `assets/` | Research visual evidence; track source and caption accurately. |
| `data/` | Readme and possibly ignored local inputs; never assume data redistribution rights. |
| `notebooks/` | Research notebooks once published and validated. |
| `docs/` | Architecture, methodology, evaluation and limitations; keep consistent with implementation. |
| `pass/` | Instructions for secret handling only; **no live secrets or credential backups**. |
| `README.md` | Reader-facing source-of-truth summary with accurate asset paths. |

## 5. Non-negotiable model/data rules

- Never run `fit`, `fit_transform`, model re-training, test-set tuning, or pseudo-label training during a user inference request.
- Never substitute a mock/random output for the shipped models in production application code. Mock models may be used in unit tests and must be clearly labeled.
- Preserve original TF-IDF model pipelines and minimal cleaning. If preprocessing changes, run regression tests against recorded input-output fixtures and explain compatibility risk.
- For single and bulk inference, handle `None`, empty strings, whitespace and `NaN` safely. Return an explicit skipped status rather than a fabricated classification.
- `LinearSVC.decision_function()` values are **not calibrated probabilities**. Only use `predict_proba()` when available, and label its output as uncalibrated unless calibration is separately established.
- Duplicate/repeated texts must not cross train/validation/test when recreating the benchmark. Avoid leakage through shared products/users/templates if a future dataset supplies identifiers.
- Do not map `Customer Rating`, sales signals or other near-target metadata into a **text-only** input without redefining the task and protocol.
- Report uncertainty honestly. Sentiment accuracy ≈95% on internal PRDECT-ID test is **not** a global marketplace accuracy guarantee. Emotion macro-F1 ≈58.86% is a documented limitation.
- Treat conflicting emotion labels as audit targets; no arbitrary auto-correction without manually reviewed evidence.

## 6. Claim boundaries (mandatory)

**Allowed:** "Sentiment macro-F1 95.07% on the recorded 811-review internal PRDECT-ID test split"; "emotion macro-F1 58.86% on the same reported test set"; "review-level text classification"; "descriptive lexical keyword analysis"; "single-review and CSV-batch Streamlit prototype".

**Forbidden without new evidence:** "95% accurate on every marketplace"; "understands customers' actual emotions"; "finds verified root causes"; "predicts churn/sales uplift"; "trained with transformers/IndoBERT"; "production-ready"; "real-time hosted SaaS"; "emotion probabilities are calibrated"; "full training is reproducible from this repo". Maintain explicit dataset, split, denominator and method qualifiers.

## 7. Engineering workflow

1. **Inspect** current Git state (`git status`, branch/head, relevant files), understand the task and find tests. Pull only if the local worktree is safe to update; don't overwrite uncommitted work.
2. **Plan** the smallest coherent change, list acceptance criteria, and identify whether it touches output contract, preprocessing, evaluation or data privacy.
3. **Implement** modularly; keep Streamlit UI thin, model/scoring logic in source modules; add docstrings/type hints for exported APIs.
4. **Validate** with targeted tests, then full tests, syntax check and a local end-to-end smoke test when possible.
5. **Report** exact commands/results. If unable to run model artifacts, say which checks remain unverified and do not claim success.
6. **Document** behavior changes in README/docs; update assets by actual tracked paths; add a clear commit message (prefer `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
7. **Commit** only authorized/safe material; avoid data files/secret leaks; never force-push or rewrite public history without explicit approval.

If changing research protocol, version the experiment instead of mutating old output artifacts. If there is a conflict between an agent's request and user-facing product safety, explain the tradeoff before proceeding.

## 8. Testing policy and required cases

Baseline command (Windows PowerShell from repository root):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m pytest -q
python -m compileall -q src app tests
```

Minimum expectations:

- preprocessing: NFKC normalization, repeated whitespace, null/NaN placeholders, Unicode/Indonesian examples, unchanged lexical semantics;
- inference: model loading from correct paths, classifier `.predict` contract, two models used, one output per input, deterministic batch behavior, empty rows not invented, decision-margin probabilities not forged;
- analytics: total/analyzed/skipped sum, distribution uses valid predictions only, no inference performed within aggregation;
- visualization: empty input, missing columns, labels, no non-available confidence fabrication; no chart calls that mutate the dataset;
- app: valid review, empty review, CSV with alternate column names, large-row limit, malformed encoding, export, missing joblib, visible error messages;
- model integration: compare known fixed input predictions under compatible versions where approved fixtures exist.

Tests currently in the repo are basic. Do not claim full coverage or successful CI without actual output. Add CI as an explicit milestone.

## 9. Research quality gate

A new model cannot replace the champion just because training took longer, accuracy increased on train, or its name is more modern. Require:

- dataset identity, license and sampling review;
- train-only transformations; no target leakage; split manifest and duplicate-group audit;
- a fair simple baseline and comparable model configurations;
- validation-only model/threshold selection;
- a truly unused final test when making new performance claims;
- macro-F1, weighted F1, accuracy, balanced accuracy, per-class precision/recall/F1 and confusion matrix;
- error analysis by negation, slang, typos, sarcasm, class imbalance, category and platform;
- reproducible seeds and dependency versions;
- no hyperparameter choices guided by a test already repeatedly inspected;
- calibration study before probability certainty claims.

If source labels or data are deficient, improve data quality first; don't disguise limitations with aggressive text cleaning, leakage or selecting the best run on a compromised test set.

## 10. Secret handling and `pass/`

The user requested a `pass/` directory. In this repository, `pass/README.md` is a **policy document only**. Never create `pass/password.txt`, `.env` with real tokens, API-key JSON, credential screenshots or private keys in tracked files. Use environment variables, GitHub Actions Secrets, a vetted password manager, or a cloud secret manager. If credentials are accidentally committed: revoke/rotate them immediately and coordinate history cleanup; deleting a file in a subsequent commit alone is insufficient.

Keep `.env.example` non-secret. Review files before commits (`git status`, `git diff --cached --stat`, search for secrets). Uploaded customer reviews can contain personal information; avoid collecting or logging them without a clear purpose, consent/legal basis, retention policy and access controls.

## 11. Model serialization and dependency safety

`joblib.load()` deserializes pickle-like content and can execute arbitrary code. Only load trusted artifacts. Verify provenance, checksums where available, and scikit-learn/joblib compatibility. Reject untrusted user-supplied models. Do not store training-only raw data in model artifacts intentionally. The current `requirements.txt` is **unpinned**; creating a validated lock file is a P0 task.

## 12. README visual evidence upkeep

When a tracked PNG changes, update the corresponding section in the root README. Current assets:

```text
assets/dataset/sentiment_distribution.png
assets/dataset/emotion_distribution.png
assets/evaluation/sentiment_test_confusion_matrix.png
assets/evaluation/emotion_test_confusion_matrix.png
assets/predictions/negative_review_top_terms.png
```

Use relative links exactly; don't point at nonexistent screenshots, untracked local plots or inferred paths. Distinguish dataset distribution from model performance. Avoid putting screenshots showing customer names or private tokens into a public repository.

## 13. Documentation, reproducibility and deployment policy

- README must always contain: problem, scope, Before–Now–Next, dataset provenance, audit, metrics, limitations, roadmap, repo tree, quick start, tests, deployment, troubleshooting, FAQ, license/attribution, and claim boundaries.
- Keep `reports/*.json` as the machine-readable evidence for **historic** results. If the model changes, add a new experiment version rather than editing history deceptively.
- Commit the training notebook and splits when license permits; add deterministic smoke fixtures and a dependency lock. State explicitly when reproducing training is impossible from remote files alone.
- Cloud deployment needs version-pinned dependencies, safe model availability, upload limits, secrets management, retention policy, monitoring and error handling. Do not invent public deployment URLs.
- License: MIT applies to original code; dataset rights and model redistribution restrictions require separate assessment.
- Ensure docs are bilingual only when actually maintained; do not imply an English/Indonesian full translation exists when it does not.

## 14. Roadmap gates (not shipped)

**P0:** training notebook + dataset revision and split manifest; lockfile; end-to-end reproducibility smoke; CI; test coverage for model contracts.

**P1:** label conflict review, new external Indonesian reviews holdout, robust emotion evaluation, UX feedback, carefully labeled issue extraction, monitoring.

**P2:** deployed service with privacy controls, CSV scale limits, versioning/rollback, documented operational metrics and a human-feedback workflow.

Check off roadmap tasks **only after** code, test output and user-facing documentation exist and are consistent.

## 15. Troubleshooting playbook for agents

| Failure | First checks |
|---|---|
| Import errors | CWD, interpreter, `PYTHONPATH=src`, requirements installed in same environment. |
| Model file missing | `models/` file names; `config.py` paths; Git LFS/push status if applicable. |
| Model deserialization error | Trusted provenance, sklearn version, joblib version, no malicious model retries. |
| Sentiment confidence blank | Expected LinearSVC behavior; never fake a confidence percent. |
| CSV upload malformed | Header, UTF-8, chosen column, row cap, null handling and corrupt bytes. |
| Metrics differ | Exact dataset revision, split IDs, label preprocessing, scoring code and dependency lock. |
| Git file ignored | `git check-ignore -v <path>`; don't force-add secrets or licensed raw data. |
| Streamlit render broken | Plotly installed, valid columns, nonempty predictions, app traceback. |
| Agent documentation request | Enumerate actual Git tree, update all paths, provide Before–Now–Next and commit details. |

## 16. Definition of done

A change is DONE only when **implemented**, **tested with recorded results**, **documented**, **safe to distribute**, and **linked to evidence**. If any part is missing, report it as partial or blocked. For model/evaluation improvements, a clear protocol plus complete metrics are mandatory; do not erase a disappointing baseline.

**Enduring project principle:** *Evidence first. Honest limitations. Human decisions remain human decisions.*
