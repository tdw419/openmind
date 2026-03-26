#!/usr/bin/env python3
"""
Glass Box Inference Engine - Real attention extraction for the spatial substrate

This script runs actual forward passes on a language model and extracts
the attention weights to drive the visual saccades.

Part of Phase 42: The Sovereign Array / Glass Box LLM
"""

import torch
import torch.nn.functional as F
import json
import sys
import numpy as np
from pathlib import Path

# Use a small model that fits in memory and has accessible attention
MODEL_NAME = "distilgpt2"  # Small, fast, has clear attention patterns

# Sentence-transformer for REAL semantic similarity (not GPT-2 hidden states)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class GlassBoxInference:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFERENCE] Loading {model_name} on {self.device}...")

        from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Load with attention outputs enabled
        config = GPT2Config.from_pretrained(model_name)
        config.output_attentions = True

        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        self.model.to(self.device)
        self.model.eval()

        # Hook storage for attention weights
        self.attention_weights = {}
        self._register_hooks()

        print(f"[INFERENCE] Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def _register_hooks(self):
        """Register forward hooks to capture attention weights"""
        def make_hook(layer_idx):
            def hook(module, input, output):
                # GPT-2 attention outputs include present key/value states
                # We want the attention weights which require special handling
                self.attention_weights[layer_idx] = {
                    'module': module.__class__.__name__,
                    'output_shape': output[0].shape if isinstance(output, tuple) else output.shape
                }
            return hook

        # Register hooks on attention modules
        for i, layer in enumerate(self.model.transformer.h):
            layer.attn.register_forward_hook(make_hook(i))

    def forward_with_attention(self, text, max_new_tokens=10):
        """
        Run a forward pass and extract attention patterns.
        Returns both the generated text and the attention data.
        """
        print(f"\n[INFERENCE] Processing: \"{text[:50]}...\"")

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        # Clear previous attention data
        self.attention_weights = {}

        # Forward pass with output_attentions=True
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )

        # Extract attention weights
        # outputs.attentions is a tuple of (batch, heads, seq, seq) for each layer
        attention_data = []
        if outputs.attentions:
            for layer_idx, attn in enumerate(outputs.attentions):
                if attn is None:
                    continue
                # attn shape: (batch, num_heads, seq_len, seq_len)
                attn_tensor = attn[0]  # Remove batch dim
                attn_np = attn_tensor.cpu().numpy()

                # Average across heads for visualization
                avg_attn = attn_np.mean(axis=0)  # (seq_len, seq_len)

                # Find the strongest attention connections
                # Each row is a token attending to previous tokens
                connections = []
                for i in range(avg_attn.shape[0]):
                    for j in range(i):  # Only causal attention (previous tokens)
                        weight = float(avg_attn[i, j])
                        if weight > 0.05:  # Threshold for visibility
                            connections.append({
                                "from_token": i,
                                "to_token": j,
                                "weight": weight
                            })

                attention_data.append({
                    "layer": layer_idx,
                    "num_heads": attn_np.shape[0],
                    "seq_len": attn_np.shape[1],
                    "top_connections": sorted(connections, key=lambda x: -x["weight"])[:20]
                })

        # Generate continuation
        generated = self.tokenizer.decode(input_ids[0])

        return {
            "input_text": text,
            "generated_text": generated,
            "num_tokens": input_len,
            "num_layers": len(attention_data),
            "attention": attention_data
        }

    def _load_embedding_model(self):
        """Lazy-load sentence-transformers for real semantic similarity."""
        if not hasattr(self, '_embed_model'):
            from sentence_transformers import SentenceTransformer
            print(f"[INFERENCE] Loading embedding model: {EMBEDDING_MODEL}")
            self._embed_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embed_model

    def compute_document_embeddings(self, archive_docs):
        """Pre-compute embeddings for all archive documents using sentence-transformers."""
        embed_model = self._load_embedding_model()
        texts = [doc.get("text", "") for doc in archive_docs]
        # Returns normalized numpy arrays
        embeddings = embed_model.encode(texts, normalize_embeddings=True)
        return embeddings  # shape: (num_docs, embed_dim)

    def compute_token_embeddings(self, tokens):
        """Compute embeddings for individual tokens using sentence-transformers."""
        embed_model = self._load_embedding_model()
        # Decode tokens to text for the embedding model
        token_texts = [self.tokenizer.decode([tid]) for tid in tokens]
        embeddings = embed_model.encode(token_texts, normalize_embeddings=True)
        return token_texts, embeddings  # (token_strings, numpy array)

    def find_closest_document(self, token_embedding, doc_embeddings):
        """Find the most semantically similar document for a token embedding."""
        if doc_embeddings is None or len(doc_embeddings) == 0:
            return 0, 0.0

        # Cosine similarity (already normalized)
        similarities = np.dot(doc_embeddings, token_embedding)
        best_idx = int(np.argmax(similarities))
        return best_idx, float(similarities[best_idx])

    def map_attention_to_saccades(self, attention_data, cortex_manifest, archive_manifest, hidden_states=None):
        """
        Map the extracted attention patterns to spatial saccades using semantic similarity.

        This connects:
        - Cortex tiles (by layer) to the attention heads
        - Archive documents to the attended tokens via embedding similarity

        Returns a list of saccade connections for visualization.
        """
        saccades = []

        cortex_tiles = cortex_manifest.get("tiles", [])
        archive_docs = archive_manifest.get("documents", [])

        if not archive_docs or not cortex_tiles:
            return saccades

        # Pre-compute document embeddings using sentence-transformers
        print(f"[INFERENCE] Computing semantic embeddings for {len(archive_docs)} archive documents...")
        doc_embeddings = self.compute_document_embeddings(archive_docs)

        # Get the actual token IDs from the input
        input_ids = self.tokenizer(attention_data["input_text"], return_tensors="pt")["input_ids"][0]
        token_texts, token_embeddings = self.compute_token_embeddings(input_ids.tolist())
        print(f"[INFERENCE] Computed embeddings for {len(token_texts)} tokens: {token_texts}")

        for layer_data in attention_data["attention"]:
            layer_idx = layer_data["layer"]

            # Find cortex tiles for this layer
            layer_tiles = [t for t in cortex_tiles
                          if f"layer_{layer_idx}" in t.get("parameter", "")]

            if not layer_tiles:
                continue

            for conn in layer_data["top_connections"]:
                # Pick a representative tile from this layer based on attention
                tile_idx = (conn["from_token"] + layer_idx) % len(layer_tiles)
                tile = layer_tiles[tile_idx]

                # SEMANTIC MATCHING: Use sentence-transformer embeddings
                doc_idx = 0
                similarity_score = 0.0

                to_tok = conn["to_token"]
                if to_tok < len(token_embeddings):
                    doc_idx, similarity_score = self.find_closest_document(
                        token_embeddings[to_tok], doc_embeddings
                    )

                # Skip connections with very low semantic similarity
                # (function words like "what", "is", "the" match nothing meaningfully)
                if similarity_score < 0.1:
                    continue

                saccades.append({
                    "tile_id": tile["id"],
                    "tile_coords": {"x": tile["x"], "y": tile["y"]},
                    "doc_id": doc_idx,
                    "doc_coords": archive_docs[doc_idx]["coords"],
                    "doc_category": archive_docs[doc_idx].get("category", "unknown"),
                    "layer": layer_idx,
                    "intensity": conn["weight"],
                    "semantic_similarity": similarity_score,
                    "from_token": conn["from_token"],
                    "from_token_text": token_texts[conn["from_token"]] if conn["from_token"] < len(token_texts) else "?",
                    "to_token": to_tok,
                    "to_token_text": token_texts[to_tok] if to_tok < len(token_texts) else "?"
                })

        return saccades


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Glass Box Inference Engine")
    parser.add_argument("--query", "-q", type=str, default="What is gravity?",
                       help="Query to process")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output JSON file for attention data")
    parser.add_argument("--export", "-e", type=str, choices=["json", "csv", "npy", "summary"],
                       default=None, help="Export format for attention data")
    parser.add_argument("--export-dir", type=str, default="exports",
                       help="Directory for exported files")
    parser.add_argument("--model", "-m", type=str, default=MODEL_NAME,
                       help="Model to use")
    args = parser.parse_args()

    # Load manifests
    project_root = Path(__file__).parent.parent
    cortex_manifest_path = project_root / "cortex" / "spatial_manifest.json"
    archive_manifest_path = project_root / "archive" / "archive_manifest.json"

    cortex_manifest = {}
    archive_manifest = {}

    if cortex_manifest_path.exists():
        cortex_manifest = json.loads(cortex_manifest_path.read_text())
    if archive_manifest_path.exists():
        archive_manifest = json.loads(archive_manifest_path.read_text())

    # Run inference
    engine = GlassBoxInference(args.model)
    result = engine.forward_with_attention(args.query)

    # Map to saccades with semantic matching (sentence-transformers)
    saccades = engine.map_attention_to_saccades(
        result, cortex_manifest, archive_manifest
    )
    result["saccades"] = saccades

    # Summary
    print(f"\n[INFERENCE] Results:")
    print(f"  Tokens processed: {result['num_tokens']}")
    print(f"  Attention layers: {result['num_layers']}")
    print(f"  Saccade connections: {len(saccades)}")

    if saccades:
        print(f"\n  Top semantic connections:")
        for s in saccades[:5]:
            print(f"    Layer {s['layer']}: Tile {s['tile_id']} → Doc {s['doc_id']} "
                  f"(sim: {s.get('semantic_similarity', 0):.3f}, attn: {s['intensity']:.3f})")

    # Output
    output_result = result
    output_path = args.output or str(project_root / "visualizations" / "real_attention.json")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output_result, indent=2))
    print(f"\n[INFERENCE] Attention data saved: {output_path}")

    # Export attention in additional formats
    if args.export:
        export_dir = project_root / args.export_dir
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.export == "json":
            # Full JSON export (already done above)
            pass

        elif args.export == "csv":
            import csv
            csv_path = export_dir / f"attention_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["layer", "from_token", "to_token", "weight", "semantic_similarity",
                                "doc_id", "doc_category", "from_token_text", "to_token_text"])
                for s in saccades:
                    writer.writerow([
                        s['layer'], s['from_token'], s['to_token'], s['intensity'],
                        s.get('semantic_similarity', 0), s['doc_id'], s['doc_category'],
                        s.get('from_token_text', ''), s.get('to_token_text', '')
                    ])
            print(f"[EXPORT] CSV saved: {csv_path}")

        elif args.export == "npy":
            # Export attention matrices as numpy arrays
            npy_dir = export_dir / f"attention_{timestamp}"
            npy_dir.mkdir(parents=True, exist_ok=True)
            for layer_data in result.get("attention", []):
                layer_idx = layer_data["layer"]
                connections = layer_data.get("top_connections", [])
                if connections:
                    # Create attention matrix from connections
                    seq_len = layer_data.get("seq_len", 10)
                    attn_matrix = np.zeros((seq_len, seq_len))
                    for conn in connections:
                        attn_matrix[conn["from_token"], conn["to_token"]] = conn["weight"]
                    np.save(npy_dir / f"layer_{layer_idx}.npy", attn_matrix)
            print(f"[EXPORT] NPY matrices saved: {npy_dir}/")

        elif args.export == "summary":
            summary_path = export_dir / f"attention_summary_{timestamp}.txt"
            with open(summary_path, 'w') as f:
                f.write(f"OpenMind Attention Summary\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Query: {args.query}\n")
                f.write(f"Model: {args.model}\n")
                f.write(f"Tokens: {result['num_tokens']}\n")
                f.write(f"Layers: {result['num_layers']}\n")
                f.write(f"Saccades: {len(saccades)}\n\n")

                f.write(f"Top Semantic Connections:\n")
                f.write(f"{'-'*50}\n")
                for s in saccades[:10]:
                    f.write(f"Layer {s['layer']}: '{s.get('from_token_text', '?')}' -> '{s.get('to_token_text', '?')}'\n")
                    f.write(f"  Doc {s['doc_id']} ({s['doc_category']}) sim={s.get('semantic_similarity', 0):.3f}\n\n")

                f.write(f"\nLayer Attention Stats:\n")
                f.write(f"{'-'*50}\n")
                for layer_data in result.get("attention", []):
                    f.write(f"Layer {layer_data['layer']}: {layer_data['num_heads']} heads, "
                           f"{len(layer_data['top_connections'])} connections\n")
            print(f"[EXPORT] Summary saved: {summary_path}")

    return result


if __name__ == "__main__":
    main()
