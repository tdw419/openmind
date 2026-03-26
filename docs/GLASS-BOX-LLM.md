# Glass Box LLM: Watching AI Think

A system that visualizes how language models process information by tracing neural attention patterns to semantically related knowledge documents.

## What It Does

Instead of treating an LLM as a black box, the Glass Box LLM lets you **watch it think**. When you ask a question, you see:

1. Which neural network weights activate (cortex tiles)
2. Which knowledge documents the model attends to (archive)
3. The semantic connections between them (saccades)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GLASS BOX LLM                                │
│                                                                      │
│   CORTEX (neural weights)          ARCHIVE (knowledge)              │
│   ┌─────────────────┐              ┌──────────────────────┐         │
│   │ ░░░░░░░░░░░░░░░ │    saccades  │ [physics]  ──────────│←──┐     │
│   │ ░░░░░░░░░░░░░░░ │ ───────────→ │ [math]              │   │     │
│   │ ░░░░░░▓▓▓▓▓░░░░ │              │ [biology]           │   │     │
│   │ ░░░░░░▓▓▓▓▓░░░░ │              │ [history]           │   │     │
│   │ ░░░░░░░░░░░░░░░ │              │ [programming] ──────│←──┤     │
│   └─────────────────┘              └──────────────────────┘   │     │
│         ↑                                                     │     │
│         │                 semantic routing                    │     │
│      active               (sentence-transformers)             │     │
│      tiles                                                 query    │
│                                                              │     │
│                         "What is gravity?" ──────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

## Architecture

### Three Core Components

| Component | What It Is | File |
|-----------|------------|------|
| **Cortex** | 79,946 tiles containing real distilgpt2 weights | `.ouroboros/spatial_llm/` |
| **Archive** | Knowledge documents with semantic embeddings | `.ouroboros/archive/` |
| **Saccades** | Attention connections between cortex and archive | Generated at runtime |

### The Cortex

The cortex is a spatial mapping of a real neural network's weights:

```
distilgpt2 (82M parameters) → Hilbert curve → 300×300 grid → 79,946 tiles
```

Each tile contains:
- 4KB of weight data (1024 float32 values)
- A spatial (x, y) coordinate on the grid
- Metadata about which layer/parameter it belongs to

**Layer types visualized:**
- Token embeddings (wte) - Blue
- Position embeddings (wpe) - Cyan
- Attention Q,K,V - Green-cyan
- Attention projection - Yellow-green
- MLP up/down - Pink/Purple
- Layer norm - Yellow

### The Archive

Knowledge documents stored with their semantic embeddings:

```json
{
  "documents": [
    {
      "id": 0,
      "category": "physics",
      "text": "Gravity is the force that pulls objects toward each other...",
      "coords": {"x": 100, "y": 200}
    }
  ]
}
```

Each document has a pre-computed embedding from `sentence-transformers/all-MiniLM-L6-v2`.

### Saccades (Attention Connections)

When you query the model, we:

1. Run the query through distilgpt2 to get **attention weights**
2. Compute **token embeddings** using sentence-transformers
3. Find the **semantically closest** archive document for each attended token
4. Create visual **saccade connections** from active cortex tiles to relevant documents

## How Semantic Routing Works

### The Problem with Random Routing

Before semantic routing, connections were based on token position:

```python
# OLD: Random routing
doc_idx = token_position % num_docs  # Token 5 → Doc 5, always
```

This meant "gravity" could route to a math document just because it was token 2.

### The Solution: Sentence-Transformer Embeddings

We use `all-MiniLM-L6-v2` for real semantic similarity:

```python
# NEW: Semantic routing
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the token
token_embedding = model.encode(["gravity"], normalize_embeddings=True)

# Embed all documents
doc_embeddings = model.encode(doc_texts, normalize_embeddings=True)

# Find closest match via cosine similarity
similarities = np.dot(doc_embeddings, token_embedding[0])
best_doc = np.argmax(similarities)  # Now correctly finds physics doc
```

### Why Not GPT-2 Hidden States?

GPT-2 wasn't trained for semantic similarity. Mean-pooled hidden states cluster everything at 0.99 cosine similarity:

```
GPT-2 hidden states:
  "gravity" → physics doc: 0.9906
  "gravity" → math doc:    0.9900
  Spread: 0.0006 (noise)

Sentence-transformers:
  "gravity" → physics doc: 0.5753
  "gravity" → math doc:    0.0560
  Spread: 0.5193 (meaningful)
```

### Filtering Noise

Function words ("what", "is", "the") have near-zero similarity to all documents. We filter them:

```python
if similarity_score < 0.1:
    continue  # Skip this connection
```

This reduces saccade count from 120 (mostly noise) to 29-38 (all meaningful).

## Usage

### Run Inference

```bash
python3 bin/inference-engine.py --query "What is gravity?"
```

Output:
- Attention data saved to `.ouroboros/visualizations/real_attention.json`
- Shows which tokens route to which documents

### Render Visualization

```bash
node bin/render-real-saccades.js
```

Output:
- PNG at `.ouroboros/visualizations/real_saccades.png`
- Visual map of cortex tiles, archive docs, and saccade lines

### Run the Executive (Map Operations)

```bash
node bin/substrate-executive.js --query "Cluster physics documents together"
```

The executive:
1. Detects intent (CLUSTER, MOVE, EXPAND, etc.)
2. Runs inference to find relevant documents
3. Executes map operations (moves document coordinates)
4. Renders decision map

## File Structure

```
.ouroboros/
├── spatial_llm/
│   ├── spatial_manifest.json    # Tile metadata
│   └── tile_*.bin               # 79,946 weight tiles
├── archive/
│   └── archive_manifest.json    # Knowledge documents
├── visualizations/
│   ├── real_cortex.png          # Weight visualization
│   ├── real_saccades.png        # Attention connections
│   └── decision_map.png         # Executive decisions
└── map_state.json               # Operation history

bin/
├── extract-real-cortex.py       # Extract distilgpt2 weights
├── inference-engine.py          # Run inference + semantic routing
├── render-real-cortex.js        # Visualize weights
├── render-real-saccades.js      # Visualize attention
└── substrate-executive.js       # Map operations
```

## Visualization Guide

### Reading the Saccade Map

```
┌────────────────────────────────────────────────────────────────┐
│  CORTEX LAYERS              │        ARCHIVE DOCUMENTS         │
│                              │                                  │
│  EMB ──░░░░░░░░░░░░░░░░░░──┐│  ┌──────────┐ ┌──────────┐      │
│  L0  ──░░░▓▓▓▓▓░░░░░░░░░──┼┼→│ physics  │ │ math     │      │
│  L1  ──░░░▓▓▓▓▓░░░░░░░░░──┘│  │ (bright) │ │ (dim)    │      │
│  L2  ──░░░░░░░░░░░░░░░░░░───│  └──────────┘ └──────────┘      │
│  ...                         │  ┌──────────┐ ┌──────────┐      │
│  L5  ──░░░░░░░░░░░░░░░░░░───│  │ biology  │ │ history  │      │
│                              │  │ (dim)    │ │ (dim)    │      │
│  ░ = inactive tile           │  └──────────┘ └──────────┘      │
│  ▓ = active (attention)      │                                  │
│                              │  bright = high semantic match    │
│  Lines = saccade connections │  dim = low/no match              │
└────────────────────────────────────────────────────────────────┘
```

### Color Coding

**Cortex layers:**
- Blue: Token embeddings (vocabulary)
- Cyan: Position embeddings
- Green/Teal: Attention mechanisms
- Pink/Purple: MLP (feedforward)
- Yellow: Layer normalization

**Archive documents:**
- Cyan: Physics
- Orange: Math
- Green: Biology
- Pink: History
- Purple: Language
- Yellow: Programming

**Saccade lines:**
- Bright cyan: High semantic similarity (>0.3)
- Dim gray: Low semantic similarity
- Thickness: Attention weight

## Example Outputs

### Query: "What is gravity?"

```
Tokens: ['What', ' is', ' gravity', ' and', ' how', ' does', ' it', ' affect', ' objects', '?']

Semantic Routing:
  ' gravity' → Doc 0 [physics] (sim: 0.575) ✓
  ' affect'  → Doc 0 [physics] (sim: 0.106) ✓
  ' objects' → Doc 8 [programming] (sim: 0.228)

Document Connections:
  Doc 0 [physics]: 27 connections (primary target)
  Doc 8 [programming]: 2 connections

Visual: Physics document glows bright, saccades converge on top-left
```

### Query: "How do cells divide in biology?"

```
Tokens: ['How', ' do', ' cells', ' divide', ' in', ' biology', '?']

Semantic Routing:
  ' cells'   → Doc 4 [biology] (sim: 0.556) ✓
  ' biology' → Doc 4 [biology] (sim: 0.382) ✓
  ' divide'  → Doc 2 [math] (sim: 0.172)

Document Connections:
  Doc 4 [biology]: 23 connections (primary target)
  Doc 2 [math]: 15 connections (secondary)

Visual: Biology document glows bright, math also active (divide=math)
```

## Extending the System

### Add New Knowledge Documents

Edit `.ouroboros/archive/archive_manifest.json`:

```json
{
  "documents": [
    {
      "id": 10,
      "category": "chemistry",
      "text": "Atoms are the basic units of matter...",
      "coords": {"x": 500, "y": 300}
    }
  ]
}
```

### Add New Map Operations

In `bin/substrate-executive.js`:

```javascript
const OPERATIONS = {
    MOVE: 'MOVE',
    CLUSTER: 'CLUSTER',
    EXPAND: 'EXPAND',
    COMPRESS: 'COMPRESS',
    ALLOCATE: 'ALLOCATE',
    SPLIT: 'SPLIT',      // New operation
    MERGE: 'MERGE'       // New operation
};

const INTENT_PATTERNS = {
    [OPERATIONS.SPLIT]: ['split', 'separate', 'divide', 'partition'],
    [OPERATIONS.MERGE]: ['merge', 'combine', 'unify', 'join']
};
```

### Use a Different Model

In `bin/inference-engine.py`:

```python
MODEL_NAME = "gpt2"  # or "gpt2-medium", "distilgpt2"
EMBEDDING_MODEL = "all-mpnet-base-v2"  # More accurate embeddings
```

## Technical Details

### Hilbert Curve Mapping

Weights are mapped to 2D space using a Hilbert curve for locality preservation:

```python
def generate_hilbert_path(n):
    """Map sequential weights to 2D grid preserving locality"""
    # Adjacent weights in memory → adjacent positions in 2D
    # This makes related parameters cluster visually
```

### Attention Extraction

GPT-2 outputs attention weights per layer when configured:

```python
config = GPT2Config.from_pretrained("distilgpt2")
config.output_attentions = True
model = GPT2LMHeadModel.from_pretrained("distilgpt2", config=config)

outputs = model(input_ids, output_attentions=True)
# outputs.attentions: tuple of (batch, heads, seq, seq) per layer
```

### Rendering Pipeline

1. Load attention JSON with saccade data
2. Create RGBA pixel buffer (1600×600)
3. Draw cortex tiles as colored dots
4. Draw archive documents as bordered boxes
5. Draw saccade lines (Bresenham's algorithm)
6. Save as raw RGBA, convert to PNG with ImageMagick

## Limitations

1. **Small archive**: 10 mock documents. Real system would have thousands.
2. **No real training data**: Documents are hand-written snippets, not actual training corpus.
3. **Static visualization**: Generates PNGs, not real-time animation.
4. **Single model**: Only distilgpt2 supported currently.
5. **No fine-tuning**: Uses pre-trained weights as-is.

## Future Directions

1. **Real training data**: Pull OpenWebText samples, UMAP-embed them into the archive
2. **Live TUI mode**: Real-time saccade animation in terminal
3. **Multi-model support**: Compare attention patterns across models
4. **Interactive exploration**: Click on tiles/docs to see details
5. **Training visualization**: Watch weights change during fine-tuning

## References

- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)
- [Sentence Transformers](https://www.sbert.net/)
- [Hilbert Curves](https://en.wikipedia.org/wiki/Hilbert_curve)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

*Part of the ASCII World project - Building transparent AI systems you can watch think.*
