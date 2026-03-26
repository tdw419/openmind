#!/usr/bin/env python3
"""
OpenMind Phase 1: Ingest training data from Wikipedia/OpenWebText/Local files.

Usage:
    python bin/ingest-training-data.py --source wikipedia --count 100
    python bin/ingest-training-data.py --source openwebtext --count 1000
    python bin/ingest-training-data.py --source local --local-dir /path/to/docs
    python bin/ingest-training-data.py --source local --local-dir ~/zion/docs/research --count 500
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import hashlib

# Lazy imports for optional dependencies
def get_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        print("Install sentence-transformers: pip install sentence-transformers")
        raise

def get_wikipedia_samples(count: int) -> List[Dict]:
    """Sample documents from Wikipedia using the datasets library."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        raise

    print(f"Loading Wikipedia dataset (sampling {count} docs)...")
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    docs = []
    for i, item in enumerate(dataset):
        if len(docs) >= count:
            break

        text = item.get('text', '')
        title = item.get('title', 'Untitled')

        # Skip very short or very long articles
        if len(text) < 200 or len(text) > 10000:
            continue

        docs.append({
            'id': len(docs),
            'title': title,
            'text': text[:2000],  # Truncate to first 2000 chars
            'source': 'wikipedia',
            'url': item.get('url', '')
        })

        if (i + 1) % 100 == 0:
            print(f"  Scanned {i + 1} articles, collected {len(docs)}")

    return docs

def get_openwebtext_samples(count: int) -> List[Dict]:
    """Sample documents from OpenWebText (Reddit links, GPT-2 training data)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        raise

    print(f"Loading OpenWebText dataset (sampling {count} docs)...")
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=True)

    docs = []
    for i, item in enumerate(dataset):
        if len(docs) >= count:
            break

        text = item.get('text', '')

        # Skip very short or very long documents
        if len(text) < 200 or len(text) > 10000:
            continue

        docs.append({
            'id': len(docs),
            'title': f"OpenWebText-{len(docs)}",
            'text': text[:2000],  # Truncate to first 2000 chars
            'source': 'openwebtext',
            'url': ''
        })

        if (i + 1) % 100 == 0:
            print(f"  Scanned {i + 1} documents, collected {len(docs)}")

    return docs

def get_tinystories_samples(count: int) -> List[Dict]:
    """Sample from TinyStories (synthetic children's stories)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Install datasets: pip install datasets")
        raise

    print(f"Loading TinyStories dataset (sampling {count} docs)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True, trust_remote_code=True)

    docs = []
    for i, item in enumerate(dataset):
        if len(docs) >= count:
            break

        text = item.get('text', '')

        if len(text) < 100:
            continue

        docs.append({
            'id': len(docs),
            'title': f"TinyStory-{len(docs)}",
            'text': text[:2000],
            'source': 'tinystories',
            'url': ''
        })

        if (i + 1) % 100 == 0:
            print(f"  Scanned {i + 1} stories, collected {len(docs)}")

    return docs

def get_local_samples(directory: str, count: int = None, extensions: List[str] = None) -> List[Dict]:
    """Load documents from local directory.

    Args:
        directory: Path to directory containing text files
        count: Maximum number of documents to load (None = all)
        extensions: File extensions to include (default: .md, .txt)
    """
    if extensions is None:
        extensions = ['.md', '.txt', '.markdown']

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    print(f"Loading local documents from {directory}...")

    docs = []
    for ext in extensions:
        for filepath in sorted(dir_path.rglob(f'*{ext}')):
            if count and len(docs) >= count:
                break

            try:
                text = filepath.read_text(encoding='utf-8', errors='ignore')
            except Exception as e:
                print(f"  Skipping {filepath.name}: {e}")
                continue

            # Skip very short or very long files
            if len(text) < 100 or len(text) > 50000:
                continue

            # Use filename as title
            title = filepath.stem.replace('_', ' ').replace('-', ' ')

            docs.append({
                'id': len(docs),
                'title': title[:100],  # Truncate long titles
                'text': text[:5000],  # Truncate to first 5000 chars
                'source': 'local',
                'url': str(filepath),
                'filename': filepath.name
            })

            if len(docs) % 100 == 0:
                print(f"  Loaded {len(docs)} documents...")

    print(f"  Total local documents loaded: {len(docs)}")
    return docs

def generate_embeddings(docs: List[Dict], model) -> np.ndarray:
    """Generate 384-dim embeddings for all documents."""
    print(f"Generating embeddings for {len(docs)} documents...")

    texts = [doc['text'] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    return np.array(embeddings)

def save_archive(docs: List[Dict], embeddings: np.ndarray, output_dir: Path):
    """Save documents and embeddings to archive."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings
    embeddings_path = output_dir / 'embeddings.npy'
    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path} (shape: {embeddings.shape})")

    # Save document manifest
    manifest = {
        'metadata': {
            'created': __import__('datetime').datetime.now().isoformat(),
            'totalDocuments': len(docs),
            'embeddingDim': embeddings.shape[1],
            'model': 'all-MiniLM-L6-v2'
        },
        'documents': docs
    }

    manifest_path = output_dir / 'archive_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description='Ingest training data for OpenMind')
    parser.add_argument('--source', choices=['wikipedia', 'openwebtext', 'tinystories', 'local'],
                        default='wikipedia', help='Data source to sample from')
    parser.add_argument('--local-dir', type=str, default=None,
                        help='Directory for local files (required if --source=local)')
    parser.add_argument('--count', type=int, default=100,
                        help='Number of documents to sample (0 or omitted for all when using local)')
    parser.add_argument('--output', type=str, default='archive',
                        help='Output directory for embeddings and manifest')

    args = parser.parse_args()

    # Get samples from selected source
    if args.source == 'wikipedia':
        docs = get_wikipedia_samples(args.count)
    elif args.source == 'openwebtext':
        docs = get_openwebtext_samples(args.count)
    elif args.source == 'tinystories':
        docs = get_tinystories_samples(args.count)
    elif args.source == 'local':
        if not args.local_dir:
            parser.error("--local-dir is required when --source=local")
        count = args.count if args.count > 0 else None
        docs = get_local_samples(args.local_dir, count=count)
    else:
        raise ValueError(f"Unknown source: {args.source}")

    print(f"Collected {len(docs)} documents from {args.source}")

    # Generate embeddings
    model = get_sentence_transformer()
    embeddings = generate_embeddings(docs, model)

    # Save to archive
    output_dir = Path(__file__).parent.parent / args.output
    save_archive(docs, embeddings, output_dir)

    print(f"\nDone! Archive ready at {output_dir}")
    print(f"  - {len(docs)} documents")
    print(f"  - {embeddings.shape[1]}-dim embeddings")
    print(f"  - Run clustering next: python bin/cluster-archive.py")

if __name__ == '__main__':
    main()
