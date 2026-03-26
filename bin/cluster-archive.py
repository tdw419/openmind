#!/usr/bin/env python3
"""
OpenMind Phase 1: Cluster archived documents using UMAP + HDBSCAN.

Usage:
    python bin/cluster-archive.py
    python bin/cluster-archive.py --neighbors 15 --min-cluster 5
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

def load_embeddings(archive_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    """Load embeddings and document manifest from archive."""
    embeddings_path = archive_dir / 'embeddings.npy'
    manifest_path = archive_dir / 'archive_manifest.json'

    if not embeddings_path.exists():
        raise FileNotFoundError(f"No embeddings found at {embeddings_path}. Run ingest-training-data.py first.")

    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest found at {manifest_path}. Run ingest-training-data.py first.")

    embeddings = np.load(embeddings_path)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Loaded {len(manifest['documents'])} documents with {embeddings.shape[1]}-dim embeddings")
    return embeddings, manifest['documents']

def reduce_dimensions(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP for spatial layout."""
    try:
        import umap
    except ImportError:
        print("Install umap-learn: pip install umap-learn")
        raise

    print(f"Reducing {embeddings.shape[0]} embeddings to 2D with UMAP...")

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='cosine',
        random_state=42
    )

    coords_2d = reducer.fit_transform(embeddings)
    print(f"UMAP projection complete. Shape: {coords_2d.shape}")

    return coords_2d

def cluster_documents(embeddings: np.ndarray, min_cluster_size: int = 5) -> np.ndarray:
    """Cluster documents using HDBSCAN."""
    try:
        import hdbscan
    except ImportError:
        print("Install hdbscan: pip install hdbscan")
        raise

    print(f"Clustering {embeddings.shape[0]} documents with HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"Found {n_clusters} clusters, {n_noise} noise points")

    return labels

def normalize_coords(coords_2d: np.ndarray, target_range: Tuple[int, int] = (0, 10000)) -> np.ndarray:
    """Normalize 2D coordinates to target range for spatial layout."""
    min_vals = coords_2d.min(axis=0)
    max_vals = coords_2d.max(axis=0)

    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    normalized = (coords_2d - min_vals) / range_vals
    scaled = normalized * (target_range[1] - target_range[0]) + target_range[0]

    return scaled.astype(int)

def get_cluster_names(labels: np.ndarray, docs: List[Dict]) -> Dict[int, str]:
    """Generate names for clusters based on common themes."""
    cluster_names = {}

    # Group docs by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(docs[i])

    # Generate names based on common words (simple approach)
    # In production, you'd use an LLM or topic modeling
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below',
                  'between', 'under', 'again', 'further', 'then', 'once', 'and',
                  'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                  'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
                  'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their',
                  'he', 'she', 'his', 'her', 'we', 'our', 'you', 'your', 'i', 'my'}

    for label, cluster_docs in clusters.items():
        # Count word frequencies
        word_freq = {}
        for doc in cluster_docs:
            words = doc['text'].lower().split()
            for word in words:
                word = ''.join(c for c in word if c.isalnum())
                if word and word not in stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Get top words
        top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:3]
        if top_words:
            cluster_names[label] = '-'.join(w[0] for w in top_words)
        else:
            cluster_names[label] = f"cluster-{label}"

    return cluster_names

def save_clustered_archive(
    docs: List[Dict],
    embeddings: np.ndarray,
    coords_2d: np.ndarray,
    labels: np.ndarray,
    archive_dir: Path
):
    """Save clustered archive with spatial coordinates."""

    # Normalize coordinates
    coords_normalized = normalize_coords(coords_2d)

    # Get cluster names
    cluster_names = get_cluster_names(labels, docs)

    # Build document records with spatial data
    documents = []
    for i, doc in enumerate(docs):
        doc_with_coords = {
            'id': doc['id'],
            'title': doc.get('title', f'Doc-{i}'),
            'text': doc['text'],
            'source': doc.get('source', 'unknown'),
            'url': doc.get('url', ''),
            'coords': {
                'x': int(coords_normalized[i][0]),
                'y': int(coords_normalized[i][1])
            },
            'cluster': int(labels[i]),
            'clusterName': cluster_names.get(labels[i], 'noise') if labels[i] != -1 else 'noise',
            'dimensions': {
                'width': 80,
                'height': min(10, max(3, len(doc['text']) // 200))
            }
        }
        documents.append(doc_with_coords)

    # Build manifest
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    manifest = {
        'metadata': {
            'created': __import__('datetime').datetime.now().isoformat(),
            'totalDocuments': len(documents),
            'totalClusters': n_clusters,
            'embeddingDim': embeddings.shape[1],
            'model': 'all-MiniLM-L6-v2',
            'layout': 'umap-2d',
            'clustering': 'hdbscan'
        },
        'clusterNames': cluster_names,
        'documents': documents
    }

    # Save manifest
    manifest_path = archive_dir / 'archive_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save 2D coordinates
    coords_path = archive_dir / 'coords_2d.npy'
    np.save(coords_path, coords_normalized)

    # Save cluster labels
    labels_path = archive_dir / 'cluster_labels.npy'
    np.save(labels_path, labels)

    print(f"\nSaved clustered archive to {archive_dir}")
    print(f"  - {len(documents)} documents")
    print(f"  - {n_clusters} clusters")
    print(f"  - Coordinates: {coords_path}")
    print(f"  - Labels: {labels_path}")

    # Print cluster summary
    print("\nCluster Summary:")
    for label in sorted(set(labels)):
        count = list(labels).count(label)
        name = cluster_names.get(label, 'noise') if label != -1 else 'noise'
        print(f"  [{label:2d}] {name}: {count} docs")

def main():
    parser = argparse.ArgumentParser(description='Cluster archived documents with UMAP + HDBSCAN')
    parser.add_argument('--archive', type=str, default='archive',
                        help='Archive directory (default: archive/)')
    parser.add_argument('--neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--min-dist', type=float, default=0.1,
                        help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--min-cluster', type=int, default=5,
                        help='HDBSCAN min_cluster_size (default: 5)')

    args = parser.parse_args()

    # Load embeddings
    archive_dir = Path(__file__).parent.parent / args.archive
    embeddings, docs = load_embeddings(archive_dir)

    # Reduce to 2D with UMAP
    coords_2d = reduce_dimensions(
        embeddings,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist
    )

    # Cluster with HDBSCAN
    labels = cluster_documents(
        embeddings,
        min_cluster_size=args.min_cluster
    )

    # Save clustered archive
    save_clustered_archive(docs, embeddings, coords_2d, labels, archive_dir)

    print("\nDone! Archive is ready for visualization.")
    print("  - Run inference: python bin/inference-engine.py")

if __name__ == '__main__':
    main()
