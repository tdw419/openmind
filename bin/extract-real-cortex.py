#!/usr/bin/env python3
"""
Real Cortex Extractor - Extracts actual distilgpt2 weights into spatial tiles

This replaces the mock weights in the cortex with real neural network parameters,
making the visualization show actual learned patterns.

Usage:
    python3 bin/extract-real-cortex.py

Output:
    .ouroboros/spatial_llm/ - Real weight tiles + updated manifest
"""

import torch
import json
import numpy as np
import os
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Config

# Configuration for distilgpt2
CONFIG = {
    "name": "DistilGPT2-Spatial",
    "model_name": "distilgpt2",
    "layers": 6,  # distilgpt2 has 6 transformer layers
    "embed_dim": 768,
    "heads": 12,
    "vocab_size": 50257,
    "tile_memory_bytes": 4096,  # 4KB per tile
    "total_tiles": 90000,  # ~82M params / 1024 floats per tile
    "grid_size": 300  # 300x300 grid
}


class RealCortexExtractor:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(__file__).parent.parent / "cortex"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Unit of data: 1 float32 = 4 bytes. 4KB tile = 1024 float32s.
        self.floats_per_tile = config["tile_memory_bytes"] // 4

    def generate_hilbert_path(self, n):
        """Hilbert curve generator for 2D mapping (locality preservation)"""
        def rot(n, x, y, rx, ry):
            if ry == 0:
                if rx == 1:
                    x = n - 1 - x
                    y = n - 1 - y
                return y, x
            return x, y

        path = []
        for i in range(n * n):
            x, y = 0, 0
            t = i
            s = 1
            while s < n:
                rx = 1 & (t // 2)
                ry = 1 & (t ^ rx)
                x, y = rot(s, x, y, rx, ry)
                x += s * rx
                y += s * ry
                t //= 4
                s *= 2
            path.append((x, y))
        return path

    def extract_weights(self):
        print(f"Loading {self.config['model_name']} from HuggingFace...")

        # Load the actual model
        model = GPT2LMHeadModel.from_pretrained(self.config["model_name"])
        state_dict = model.state_dict()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params:,} parameters ({total_params * 4 / 1024 / 1024:.1f} MB)")

        manifest = {
            "metadata": {
                **self.config,
                "total_parameters": total_params,
                "extraction_date": str(np.datetime64('now'))
            },
            "tiles": [],
            "layer_stats": {}
        }

        # Hilbert path for mapping tile IDs to 2D coordinates
        print(f"Generating Hilbert path for {self.config['grid_size']}x{self.config['grid_size']} grid...")
        hilbert_path = self.generate_hilbert_path(self.config["grid_size"])

        current_tile_id = 0

        def process_tensor(name, tensor):
            nonlocal current_tile_id

            flat_data = tensor.detach().cpu().numpy().flatten().astype(np.float32)
            num_tiles = (len(flat_data) + self.floats_per_tile - 1) // self.floats_per_tile

            # Calculate stats for this parameter
            stats = {
                "shape": list(tensor.shape),
                "num_params": len(flat_data),
                "num_tiles": num_tiles,
                "mean": float(np.mean(flat_data)),
                "std": float(np.std(flat_data)),
                "min": float(np.min(flat_data)),
                "max": float(np.max(flat_data))
            }
            manifest["layer_stats"][name] = stats

            print(f"  Mapping {name}: {len(flat_data):,} params -> {num_tiles} tiles "
                  f"(μ={stats['mean']:.4f}, σ={stats['std']:.4f})")

            for i in range(num_tiles):
                start = i * self.floats_per_tile
                end = min(start + self.floats_per_tile, len(flat_data))
                chunk = flat_data[start:end]

                # Zero-pad if needed
                if len(chunk) < self.floats_per_tile:
                    chunk = np.pad(chunk, (0, self.floats_per_tile - len(chunk)))

                # Save chunk to disk
                chunk_filename = f"tile_{current_tile_id}.bin"
                with open(self.output_dir / chunk_filename, "wb") as f:
                    f.write(chunk.tobytes())

                # Map tile ID to 2D coordinate via Hilbert path
                if current_tile_id < len(hilbert_path):
                    coords = hilbert_path[current_tile_id]
                else:
                    # Fallback to linear if we exceed grid size
                    grid_size = self.config["grid_size"]
                    coords = (current_tile_id % grid_size, current_tile_id // grid_size)

                manifest["tiles"].append({
                    "id": current_tile_id,
                    "x": coords[0],
                    "y": coords[1],
                    "parameter": name,
                    "offset": i,
                    "file": chunk_filename,
                    "stats": {
                        "chunk_mean": float(np.mean(chunk)),
                        "chunk_std": float(np.std(chunk))
                    }
                })

                current_tile_id += 1

        # Process all weights in a logical order
        print("\nExtracting weights:")

        # 1. Token embeddings (vocabulary)
        process_tensor("wte.weight", state_dict["transformer.wte.weight"])

        # 2. Position embeddings
        process_tensor("wpe.weight", state_dict["transformer.wpe.weight"])

        # 3. Transformer layers
        for i in range(self.config["layers"]):
            prefix = f"transformer.h.{i}"

            # Self-attention
            process_tensor(f"layer_{i}_attn_c_attn", state_dict[f"{prefix}.attn.c_attn.weight"])
            process_tensor(f"layer_{i}_attn_c_proj", state_dict[f"{prefix}.attn.c_proj.weight"])

            # Layer norms
            process_tensor(f"layer_{i}_ln_1", state_dict[f"{prefix}.ln_1.weight"])
            process_tensor(f"layer_{i}_ln_2", state_dict[f"{prefix}.ln_2.weight"])

            # MLP
            process_tensor(f"layer_{i}_mlp_c_fc", state_dict[f"{prefix}.mlp.c_fc.weight"])
            process_tensor(f"layer_{i}_mlp_c_proj", state_dict[f"{prefix}.mlp.c_proj.weight"])

        # 4. Final layer norm
        process_tensor("ln_f.weight", state_dict["transformer.ln_f.weight"])

        # 5. Output projection (tied with embeddings in distilgpt2, but still present)
        # Note: In distilgpt2, lm_head weight is tied to wte, so we skip it

        # Write the manifest
        manifest["metadata"]["total_tiles"] = current_tile_id

        with open(self.output_dir / "spatial_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"\n✅ Extraction complete!")
        print(f"   Total tiles: {current_tile_id:,}")
        print(f"   Grid coverage: {current_tile_id / (self.config['grid_size'] ** 2) * 100:.1f}%")
        print(f"   Manifest: {self.output_dir}/spatial_manifest.json")

        return manifest


if __name__ == "__main__":
    extractor = RealCortexExtractor(CONFIG)
    manifest = extractor.extract_weights()

    # Print summary
    print("\n📊 Layer Summary:")
    for name, stats in manifest["layer_stats"].items():
        print(f"   {name}: {stats['num_params']:,} params, μ={stats['mean']:.4f}, σ={stats['std']:.4f}")
