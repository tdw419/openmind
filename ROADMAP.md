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

**Success Criteria:**
- [ ] 1,000+ documents in archive
- [ ] Clustering produces 10+ topic clusters
- [ ] Query latency < 500ms with full archive
- [ ] Archive manifest generated automatically

**Risks:**
- Embedding 1000 docs may take hours → use batch processing
- UMAP projection may not be meaningful → test with known clusters first

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

**Success Criteria:**
- [ ] Token-by-token saccade animation working
- [ ] < 100ms latency per token
- [ ] TUI updates smoothly (30+ fps)
- [ ] Streaming works with all sentence-transformer models

**Risks:**
- Python GIL may limit streaming speed → consider multiprocessing
- Attention extraction may slow inference → cache intermediate results

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

**Success Criteria:**
- [ ] 3+ models supported (distilgpt2, gpt2, TinyLlama)
- [ ] Side-by-side TUI comparison working
- [ ] Model switching without restart
- [ ] Attention diff highlighting

**Risks:**
- Different architectures may need custom attention extraction
- Large models (Phi-2) may be slow on CPU → require GPU

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

**Success Criteria:**
- [ ] Cortex visualization in browser (clickable, zoomable)
- [ ] Document panel shows full text + embedding
- [ ] WebSocket streaming at 30+ fps
- [ ] Works in Chrome, Firefox, Safari

**Risks:**
- D3.js performance with 80k tiles → use WebGL or canvas
- WebSocket complexity → use Socket.io or similar library

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

**Success Criteria:**
- [ ] Multi-turn conversation history stored
- [ ] Memory heatmap visualization
- [ ] Context window boundary shown
- [ ] Attention comparison across turns

**Risks:**
- Context window limits may truncate history → implement sliding window
- Memory visualization may be cluttered → add filtering options

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

**Success Criteria:**
- [ ] Weight snapshots captured during training
- [ ] Before/after cortex comparison
- [ ] Diff view showing changed tiles
- [ ] Animation of weight evolution

**Risks:**
- Snapshot storage may be large → compress or sample
- LoRA extraction may need custom implementation
- Training may crash → implement checkpoint recovery

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

**Success Criteria:**
- [ ] 5+ guided lessons implemented
- [ ] Interactive exercises working
- [ ] Quiz mode with scoring
- [ ] Plain-language explanations for all visualizations

**Risks:**
- Content creation is time-consuming → use LLM to draft lessons
- Quiz answers may be ambiguous → test with real users

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

**Success Criteria:**
- [ ] Export to CSV/JSON for statistical analysis
- [ ] Attention head decomposition working
- [ ] Cross-language comparison supported
- [ ] Research paper citation ready

**Risks:**
- Academic adoption may be slow → publish blog posts first
- Research features may not match user needs → survey researchers

---

## Version Milestones

| Version | Phases | Deliverable | Target |
|---------|--------|-------------|--------|
| v1.1 | 1 + 2 | Real data + streaming TUI | 1 week |
| v1.5 | 3 + 4 | Multi-model + web dashboard | 3 weeks |
| v2.0 | 5 + 6 | Conversation + fine-tuning viz | 6 weeks |
| v2.5 | 7 + 8 | Educational + research tools | Ongoing |

## Dependencies

```
v1.0 ──► Phase 1 (Real Data) ──► Phase 2 (Streaming)
                                   │
                                   ▼
                     Phase 4 (Web Dashboard) ──► Phase 3 (Multi-Model)
                                                   │
                                                   ▼
                                 Phase 5 (Conversation) ──► Phase 6 (Fine-Tuning)
                                                                │
                                                                ▼
                                              Phase 7 (Education) ◄── Phase 8 (Research)
```

**Critical Path:** 1 → 2 → 4 → 5 (longest chain to v2.0)

## Priority Matrix

| Phase | Effort | Impact | Dependencies | Priority Score |
|-------|--------|--------|--------------|----------------|
| 1. Real Data | 2d | HIGH | None | **9.0** ⭐ |
| 2. Streaming | 4d | HIGH | Phase 1 | **8.5** ⭐ |
| 4. Web Dashboard | 2w | HIGH | Phase 2 | **7.5** |
| 3. Multi-Model | 5d | MED | Phase 2 | **6.5** |
| 5. Conversation | 1w | MED | Phase 4 | **6.0** |
| 6. Fine-Tuning | 2w | MED | Phase 5 | **5.0** |
| 7. Education | 2w | MED | Phase 6 | **4.5** |
| 8. Research | Ongoing | LOW | Phase 7 | **3.0** |

## Not in Scope (Explicit Boundaries)

| Feature | Why Excluded |
|---------|--------------|
| Model training from scratch | Focus is visualization, not training |
| Production API server | Research tool, not production service |
| Mobile app | Desktop-first, web dashboard covers mobile |
| Real-time collaboration | Single-user tool for now |
| Plugin system | Premature optimization |
| GPU cluster support | Single-machine focus |

## Technical Debt to Address

| Debt | Phase to Address | Fix |
|------|------------------|-----|
| Mock archive (10 docs) | Phase 1 | Replace with real data |
| Batch inference only | Phase 2 | Add streaming |
| Single model hardcoded | Phase 3 | Multi-model architecture |
| No persistence | Phase 4 | Add database layer |
| TUI only | Phase 4 | Web dashboard |
| No tests | Phase 1 | Add pytest suite |

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
