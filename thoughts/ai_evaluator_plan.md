# AI Evaluator Project - 4 Week Plan

## Goal
Build a sellable AI Evaluator / AI QA framework that tests LLM outputs, scores them, catches regressions, generates reports.

## Skills Focus
- HuggingFace transformers (Week 2)
- Fine-tuning (Week 2-3)
- CLI design (Week 1)
- System architecture (All weeks)
- LLM integration (Week 4)
- Regression detection & metrics (Week 3)

## Timeline
- **Week 1:** Evaluator skeleton + schemas (CLI, file formats, repo structure)
- **Week 2:** Classification model (HF) + deterministic scoring (real end-to-end)
- **Week 3:** Regression detection + model comparison (catch degradation)
- **Week 4:** LLM-judge + demo suites + B2B packaging (make it sellable)

## Current Status
- ✅ Folder structure created locally
- ✅ `/Users/macski/claude/ai-evaluator-project/` ready
- Starting Week 1.1 - Schema definition

## Key Deliverables by Week

### Week 1
- [ ] Testcase schema (JSON/YAML format)
- [ ] Model config schema
- [ ] CLI skeleton: `eval run --suite suites/basic --config configs/model_a.yaml`
- [ ] Sample 10 test cases
- [ ] Repo structure finalized
- [ ] README with product definition

### Week 2
- [ ] LLM client wrapper (provider-agnostic)
- [ ] Deterministic scorers (JSON schema, regex, formatting)
- [ ] HuggingFace classification model integrated
- [ ] End-to-end: prompt → model → score → report
- [ ] Markdown + JSON reporting

### Week 3
- [ ] Baseline storage system
- [ ] `eval compare` command (run vs baseline)
- [ ] Regression gates (fail if score drops > X%)
- [ ] Statistical analysis

### Week 4
- [ ] LLM-judge scorer (Claude/GPT rubric-based)
- [ ] Multiple demo suites (sentiment, credit risk, etc.)
- [ ] HTML report export (optional)
- [ ] Sales one-pager in README
- [ ] Deployment ready

## Market Target
- Companies using internal LLMs / copilots
- Teams shipping LLM features
- Compliance-minded organizations
- Offer: Pilots, retainers, CI/CD integration

## Next Action
Start Week 1.1: Define schemas (testcase + config formats)
