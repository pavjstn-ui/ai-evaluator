# AI Evaluator - Schema Documentation

## Overview
Two core schemas define the AI Evaluator system:
1. **TestCase Schema** — What gets evaluated
2. **Config Schema** — How the model is configured

---

## TestCase Schema

### Purpose
A testcase is a single evaluation unit. It contains an input, expected output, and scoring rubric.

### Format
YAML or JSON (recommended: YAML for readability)

### Fields

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `id` | string | ✅ YES | Unique identifier for tracking | `"sentiment_001"` |
| `task` | string | ✅ YES | Instruction given to model | `"Classify sentiment as POSITIVE/NEGATIVE/NEUTRAL"` |
| `input` | string | ✅ YES | The actual text to evaluate | `"Výborný produkt!"` |
| `context` | string | ⬜ NO | Additional context (optional system prompt) | `"You are a sentiment classifier..."` |
| `expected` | string/object | ✅ YES | Expected output | `"POSITIVE"` |
| `rubric` | object | ✅ YES | Scoring criteria with weights | See below |
| `tags` | list | ✅ YES | Categorization for filtering | `["domain:sentiment", "language:czech"]` |

### Rubric Structure

```yaml
rubric:
  criterion_name:
    description: "What this measures"
    weight: 0.7  # 0-1, sum of all weights should = 1.0
    rule: "scoring_rule"  # See scoring rules below
  criterion_name_2:
    description: "..."
    weight: 0.3
    rule: "..."
```

### Scoring Rules

| Rule | Type | Description | Example |
|------|------|-------------|---------|
| `exact_match` | Deterministic | Output must match `expected` exactly | Labels: POSITIVE, NEGATIVE |
| `score_above_X` | Deterministic | Confidence score must be > X | `score_above_0.85` |
| `json_valid` | Deterministic | Output must be valid JSON | APIs returning JSON |
| `required_keys` | Deterministic | JSON must contain specific keys | `["name", "score"]` |
| `forbidden_phrases` | Deterministic | Output must NOT contain phrases | `["password", "credit_card"]` |
| `length_max_X` | Deterministic | Output length ≤ X chars | `length_max_500` |
| `rubric_score_X_to_Y` | LLM-Judge | LLM scores output on scale X-Y | `rubric_score_0_to_5` |

### Tag Format

Tags use `key:value` format for filtering:

```yaml
tags:
  - domain: "sentiment"           # What problem domain
  - language: "czech"              # What language
  - difficulty: "easy|medium|hard" # Complexity
  - risk_level: "low|medium|high"  # Business impact if wrong
```

### Complete TestCase Example

```yaml
id: "sentiment_004"
task: "Classify the sentiment of this Czech review as POSITIVE, NEGATIVE, or NEUTRAL."
input: "Skvělé! Vypadá to, že to není vůbec padělané."
expected: "NEGATIVE"

context: |
  You are a sentiment classifier for Czech product reviews.
  Be careful with sarcasm and implicit meanings.
  Return ONLY the label in uppercase.

rubric:
  accuracy:
    description: "Is the classification correct?"
    weight: 0.8
    rule: "exact_match"
  confidence:
    description: "Is model sufficiently confident?"
    weight: 0.2
    rule: "score_above_0.7"

tags:
  - domain: "sentiment"
  - language: "czech"
  - difficulty: "hard"
  - risk_level: "medium"
```

---

## Config Schema

### Purpose
A config defines how a specific model/prompt combination should be run.

### Format
YAML recommended

### Fields

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `name` | string | ✅ YES | Config identifier | `"czech-roberta-v1"` |
| `provider` | string | ✅ YES | Model provider | `"huggingface"`, `"openai"`, `"anthropic"` |
| `model` | string | ✅ YES | Model ID/name | `"xlm-roberta-base"` |
| `task_type` | string | ⬜ NO | Task (for HF) | `"text-classification"` |
| `system_prompt` | string | ⬜ NO | System/instruction prefix | `"You are..."` |
| `temperature` | float | ⬜ NO | Sampling temperature (0.0-1.0) | `0.3` |
| `max_tokens` | int | ⬜ NO | Max output length | `50` |
| `batch_size` | int | ⬜ NO | Batch size for inference | `8` |
| `seed` | int | ⬜ NO | Random seed for reproducibility | `42` |
| `num_labels` | int | ⬜ NO | Number of output classes (HF) | `3` |
| `fine_tuned_weights` | string | ⬜ NO | Path to fine-tuned weights | `"./models/fine_tuned_v1"` |

### Provider-Specific Fields

#### HuggingFace
```yaml
provider: "huggingface"
model: "xlm-roberta-base"
task_type: "text-classification"
num_labels: 3
fine_tuned_weights: null  # or path
```

#### OpenAI
```yaml
provider: "openai"
model: "gpt-4"
system_prompt: "You are a sentiment classifier..."
temperature: 0.3
max_tokens: 50
```

#### Anthropic
```yaml
provider: "anthropic"
model: "claude-3-sonnet"
system_prompt: "You are a sentiment classifier..."
temperature: 0.3
max_tokens: 50
```

### Complete Config Examples

**HuggingFace - Baseline:**
```yaml
name: "xlm-roberta-baseline"
provider: "huggingface"
model: "xlm-roberta-base"
task_type: "text-classification"
num_labels: 3
seed: 42
temperature: 0.3
batch_size: 8
```

**HuggingFace - Fine-Tuned:**
```yaml
name: "xlm-roberta-czech-finetuned"
provider: "huggingface"
model: "xlm-roberta-base"
fine_tuned_weights: "./models/czech_sentiment_model"
task_type: "text-classification"
num_labels: 3
seed: 42
temperature: 0.1
batch_size: 8
```

**Claude LLM-Judge:**
```yaml
name: "claude-judge-rubric"
provider: "anthropic"
model: "claude-3-sonnet"
system_prompt: |
  You are an expert evaluator of LLM outputs.
  Score responses on the provided rubric.
  Return JSON with scores and reasoning.
temperature: 0.3
max_tokens: 500
```

---

## File Organization

```
ai-evaluator/
├── suites/
│   ├── basic/
│   │   ├── 001_sentiment_positive.yaml
│   │   ├── 002_sentiment_negative.yaml
│   │   └── ...
│   ├── credit_risk/
│   │   ├── 001_low_risk.yaml
│   │   ├── 002_high_risk.yaml
│   │   └── ...
│   └── customer_support/
│       ├── 001_helpful_response.yaml
│       └── ...
├── configs/
│   ├── xlm_roberta_baseline.yaml
│   ├── xlm_roberta_finetuned.yaml
│   ├── claude_judge.yaml
│   └── gpt4_judge.yaml
└── baselines/
    ├── basic/
    │   ├── xlm_roberta_baseline.json
    │   └── xlm_roberta_finetuned.json
    └── credit_risk/
        ├── xlm_roberta_baseline.json
        └── xlm_roberta_finetuned.json
```

---

## Next Steps

1. ✅ Define schemas (done)
2. Create 10-20 realistic test cases
3. Define model configs
4. Build CLI to load and run them
