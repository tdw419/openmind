# OpenMind

Watch an AI think. OpenMind visualizes how language models process information by tracing neural attention patterns to semantically related knowledge.

![Saccade Visualization](docs/saccades.png)

## What It Does

When you ask a question, OpenMind shows you:

1. **Which neural weights activate** — 79,946 tiles from distilgpt2
2. **Which knowledge the model attends to** — semantic routing to documents
3. **The connections between them** — saccade lines tracing attention

Query "What is gravity?" → watch saccades converge on physics documents.
Query "How do cells divide?" → watch them shift to biology.

## Quick Start

```bash
# Clone
git clone https://github.com/yourname/openmind.git
cd openmind

# Install dependencies
pip install -r requirements.txt
npm install

# Run inference
python3 bin/inference-engine.py --query "What is gravity?"

# Visualize
node bin/render-real-saccades.js
```

Output: `visualizations/real_saccades.png`

## How It Works

### Semantic Routing

The key innovation is semantic saccade routing using sentence-transformers:

```python
# OLD: Random routing
doc_idx = token_position % num_docs  # "gravity" → random doc

# NEW: Semantic routing
token_embedding = sentence_transformer.encode("gravity")
doc_idx = argmax(cosine_similarity(token_embedding, doc_embeddings))
# "gravity" → physics document ✓
```

This replaces the `token % num_docs` hack with actual semantic similarity, making the attention traces meaningful.

### Components

| Component | Description |
|-----------|-------------|
| **Cortex** | 79,946 tiles (4KB each) containing distilgpt2 weights, mapped via Hilbert curve |
| **Archive** | Knowledge documents with pre-computed embeddings |
| **Saccades** | Attention connections routed by semantic similarity |

## Commands

```bash
# Run inference on a query
python3 bin/inference-engine.py --query "Your question here"

# Render the cortex (neural weight visualization)
node bin/render-real-cortex.js

# Render saccades (attention connections)
node bin/render-real-saccades.js

# Run the substrate executive (map operations)
node bin/substrate-executive.js --query "Cluster physics documents together"
```

## Architecture

```
openmind/
├── cortex/               # 79,946 weight tiles from distilgpt2
│   ├── spatial_manifest.json
│   └── tile_*.bin       # 4KB chunks of float32 weights
├── archive/              # Knowledge documents
│   └── archive_manifest.json
├── visualizations/       # Output PNGs
├── bin/
│   ├── extract-real-cortex.py    # Extract weights from model
│   ├── inference-engine.py       # Run inference + semantic routing
│   ├── render-real-cortex.js     # Visualize weights
│   ├── render-real-saccades.js   # Visualize attention
│   └── substrate-executive.js    # Map operations
└── docs/
    └── GLASS-BOX-LLM.md          # Full documentation
```

## Requirements

- Python 3.8+ with PyTorch, transformers, sentence-transformers
- Node.js 16+ (for rendering)
- ImageMagick (for PNG conversion)

## Full Documentation

See [docs/GLASS-BOX-LLM.md](docs/GLASS-BOX-LLM.md) for:

- Detailed architecture explanation
- How semantic routing works (and why not GPT-2 hidden states)
- Visualization guide
- Extending the system
- Technical implementation details

## License

MIT

---

*Part of the ASCII World project — Building transparent AI systems you can watch think.*
