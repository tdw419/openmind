# OpenMind Roadmap

## Current State (v1.0)

| Component | Status | Notes |
|-----------|--------|-------|
| Cortex extraction | ✅ Done | 79,946 tiles from distilgpt2 |
| Semantic routing | ✅ Done | sentence-transformers, not GPT-2 hidden states |
| PNG visualization | ✅ Done | Static saccade maps |
| TUI prototype | ✅ Done | Live terminal with bloom effect |
| Archive | ⚠️ Mock | 10 hand-written documents |
| Inference | ⚠️ Batch | Run query → wait → see results |

---

## Phase 1: Real Training Data

**Goal:** Replace 10 mock documents with 1,000+ real samples from training corpora.

### 1.1 Data Sources

| Source | Size | Content |
|--------|------|---------|
| OpenWebText | 8M docs | Reddit links (GPT-2 training data) |
| TinyStories | 2M docs | Synthetic children's stories |
| Wikipedia | 6M docs | Encyclopedia articles |
| ArXiv | 2M docs | Scientific papers |

### 1.2 Implementation

```
1. Sample 1000 docs from OpenWebText
2. Embed with sentence-transformers (384-dim)
3. UMAP project to 2D for spatial layout
4. Cluster by topic (HDBSCAN)
5. Generate archive_manifest.json
```

### 1.3 Files to Create

- `bin/ingest-training-data.py` — Sample + embed corpus
- `bin/cluster-archive.py` — UMAP + HDBSCAN clustering
- `archive/embeddings.npy` — Pre-computed embeddings

**Effort:** 2-3 days
**Impact:** Makes semantic routing meaningful at scale

---

## Phase 2: Streaming Attention

**Goal:** Watch attention shift token-by-token during generation.

### 2.1 Architecture

```
Current:
  Query → Full inference → All saccades at once

Target:
  Query → Token 1 → Saccades for token 1
        → Token 2 → Saccades for token 2
        → Token 3 → ...
```

### 2.2 Implementation

```python
# In inference-engine.py
def stream_inference(query):
    tokens = tokenize(query)
    for i, token in enumerate(tokens):
        # Get attention for this token only
        attention = get_token_attention(token, previous_tokens)
        yield {
            "token": token,
            "position": i,
            "saccades": compute_saccades(attention)
        }
```

### 2.3 TUI Integration

- Animate saccades as each token streams in
- Show "thinking" cursor moving through cortex
- Update document attention bars in real-time

**Effort:** 3-4 days
**Impact:** See the AI "think" in real-time

---

## Phase 3: Multi-Model Support

**Goal:** Compare attention patterns across different models.

### 3.1 Models to Support

| Model | Params | Why |
|-------|--------|-----|
| distilgpt2 | 82M | Current baseline |
| gpt2 | 124M | Original GPT-2 |
| gpt2-medium | 355M | Deeper attention |
| TinyLlama | 1.1B | Modern architecture |
| Phi-2 | 2.7B | High quality small model |

### 3.2 Implementation

```
bin/
├── extract-cortex.py --model gpt2
├── inference-engine.py --model tinyllama
└── compare-models.py  # Side-by-side visualization
```

### 3.3 Visualization

- Split-screen TUI showing two models side-by-side
- Highlight differences in attention patterns
- "Why does model A attend here while model B attends there?"

**Effort:** 4-5 days
**Impact:** Compare model "thinking styles"

---

## Phase 4: Interactive Web Dashboard

**Goal:** Browser-based visualization with clickable exploration.

### 4.1 Features

- [ ] Clickable cortex tiles → see weight statistics
- [ ] Clickable documents → see full text + embeddings
- [ ] Drag to pan/zoom cortex visualization
- [ ] Hover over saccade lines → see similarity score
- [ ] Query input with live suggestions

### 4.2 Tech Stack

```
Frontend:  React + D3.js or WebGL
Backend:   FastAPI serving cortex/archive
Real-time: WebSocket for streaming attention
```

### 4.3 File Structure

```
openmind/
├── web/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CortexGrid.tsx
│   │   │   ├── ArchivePanel.tsx
│   │   │   └── SaccadeCanvas.tsx
│   │   └── App.tsx
│   └── package.json
├── server/
│   └── api.py
```

**Effort:** 1-2 weeks
**Impact:** Accessible to non-technical users

---

## Phase 5: Conversation Mode

**Goal:** Visualize attention across multi-turn conversations.

### 5.1 Features

- Conversation history stored in context
- Show how attention shifts based on previous turns
- Highlight "memory" — when model attends to earlier context

### 5.2 Implementation

```
class ConversationContext:
    def __init__(self):
        self.history = []
        self.attention_history = []

    def add_turn(self, query, response, attention):
        self.history.append((query, response))
        self.attention_history.append(attention)

    def get_context_attention(self):
        # How much does current query attend to previous context?
        pass
```

### 5.3 Visualization

- Timeline view showing attention across turns
- "Memory heatmap" — which parts of conversation are most attended to
- Context window boundary visualization

**Effort:** 1 week
**Impact:** Understand context usage in LLMs

---

## Phase 6: Fine-Tuning Visualization

**Goal:** Watch weights change during fine-tuning.

### 6.1 Features

- Snapshot cortex before/after fine-tuning
- Animate weight changes over training steps
- Highlight which tiles changed most
- Compare LoRA adapters vs full fine-tuning

### 6.2 Implementation

```python
# Capture weight snapshots during training
for step in training_loop:
    if step % 100 == 0:
        save_cortex_snapshot(step)

# Render animation
render_weight_evolution(snapshots)
```

### 6.3 Visualization

- Side-by-side before/after cortex
- "Diff" view showing changed tiles
- Animation of weight evolution over time

**Effort:** 2 weeks
**Impact:** Understand how training shapes attention

---

## Phase 7: Educational Mode

**Goal:** Teaching tool for understanding LLMs.

### 7.1 Features

- Guided tours: "How does attention work?"
- Interactive exercises: "Predict where attention will focus"
- Quiz mode: "Which document will this query attend to?"
- Explanations in plain language

### 7.2 Content

```
lessons/
├── 01-what-is-attention.md
├── 02-semantic-similarity.md
├── 03-token-embeddings.md
├── 04-layer-by-layer.md
└── 05-attention-heads.md
```

**Effort:** 1-2 weeks
**Impact:** Educational value, wider audience

---

## Phase 8: Research Integration

**Goal:** Tool for attention mechanism research.

### 8.1 Features

- Export attention patterns for analysis
- Statistical summaries across queries
- Attention head decomposition
- Compare attention patterns across languages

### 8.2 Research Questions

- Do certain attention patterns correlate with hallucination?
- How does attention differ for factual vs creative queries?
- What attention patterns indicate uncertainty?

**Effort:** Ongoing
**Impact:** Academic citations, research collaborations

---

## Priority Matrix

```
                    High Impact
                        │
         Phase 1 ●      │      ● Phase 2
         (Real Data)    │      (Streaming)
                        │
    Phase 7 ●           │           ● Phase 4
    (Education)         │           (Web Dashboard)
                        │
Low Effort ─────────────┼───────────── High Effort
                        │
         Phase 3 ●      │      ● Phase 5
         (Multi-Model)  │      (Conversation)
                        │
         Phase 8 ●      │      ● Phase 6
         (Research)     │      (Fine-Tuning)
                        │
                    Low Impact
```

## Recommended Order

1. **Phase 1** — Real Training Data (quick win, high impact)
2. **Phase 2** — Streaming Attention (core experience)
3. **Phase 4** — Web Dashboard (accessibility)
4. **Phase 3** — Multi-Model (research value)
5. **Phase 5** — Conversation Mode (advanced feature)
6. **Phase 6** — Fine-Tuning (specialized use case)
7. **Phase 7** — Education (polish)
8. **Phase 8** — Research (ongoing)

---

## Quick Wins (This Week)

- [ ] Add 100 real documents from Wikipedia
- [ ] Animate saccade lines in TUI (not just static)
- [ ] Add "export attention" command
- [ ] Color-code attention by layer in visualization
- [ ] Add token position indicator in TUI

## Metrics to Track

| Metric | Current | Target (Phase 1) |
|--------|---------|------------------|
| Archive size | 10 | 1,000+ |
| Inference latency | 2-3s | <1s (streaming) |
| TUI frame rate | 10 fps | 30 fps |
| Model support | 1 | 5+ |
| GitHub stars | — | 1,000 |

---

*Last updated: 2026-03-26*
