#!/usr/bin/env python3
"""
OpenMind Phase 1 Tests

Tests for data ingestion and clustering scripts.

Run with: pytest tests/test_phase1.py -v
"""

import pytest
import json
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add bin to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'bin'))


class TestIngestTrainingData:
    """Tests for ingest-training-data.py functionality."""

    def test_get_tinystories_samples(self):
        """Test TinyStories sampling (smallest dataset for fast tests)."""
        # Import the function
        from ingest_training_data import get_tinystories_samples

        # Sample just 5 docs for quick test
        docs = get_tinystories_samples(5)

        assert len(docs) == 5
        for doc in docs:
            assert 'id' in doc
            assert 'text' in doc
            assert 'source' in doc
            assert doc['source'] == 'tinystories'
            assert len(doc['text']) >= 100

    def test_generate_embeddings_shape(self):
        """Test embedding generation produces correct shape."""
        from ingest_training_data import generate_embeddings, get_sentence_transformer

        docs = [
            {'id': 0, 'text': 'This is a test document about physics.'},
            {'id': 1, 'text': 'Mathematics is the study of numbers.'},
            {'id': 2, 'text': 'Biology studies living organisms.'}
        ]

        model = get_sentence_transformer()
        embeddings = generate_embeddings(docs, model)

        assert embeddings.shape[0] == 3  # 3 docs
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 dimension

    def test_save_archive(self):
        """Test archive saving creates correct files."""
        from ingest_training_data import save_archive

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            docs = [
                {'id': 0, 'text': 'Test document 1', 'source': 'test'},
                {'id': 1, 'text': 'Test document 2', 'source': 'test'}
            ]
            embeddings = np.random.rand(2, 384).astype(np.float32)

            save_archive(docs, embeddings, output_dir)

            # Check files exist
            assert (output_dir / 'embeddings.npy').exists()
            assert (output_dir / 'archive_manifest.json').exists()

            # Check manifest content
            with open(output_dir / 'archive_manifest.json') as f:
                manifest = json.load(f)

            assert manifest['metadata']['totalDocuments'] == 2
            assert manifest['metadata']['embeddingDim'] == 384
            assert len(manifest['documents']) == 2


class TestClusterArchive:
    """Tests for cluster-archive.py functionality."""

    def test_normalize_coords(self):
        """Test coordinate normalization."""
        from cluster_archive import normalize_coords

        coords = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ])

        normalized = normalize_coords(coords, target_range=(0, 1000))

        assert normalized[0, 0] == 0
        assert normalized[0, 1] == 0
        assert normalized[1, 0] == 1000
        assert normalized[1, 1] == 1000
        assert 400 <= normalized[2, 0] <= 600

    def test_cluster_names_generation(self):
        """Test cluster name generation from documents."""
        from cluster_archive import get_cluster_names

        labels = np.array([0, 0, 1, 1, -1])
        docs = [
            {'text': 'The cat sat on the mat and purred.'},
            {'text': 'Dogs are loyal pets that bark.'},
            {'text': 'Python is a programming language.'},
            {'text': 'JavaScript runs in the browser.'},
            {'text': 'This is noise.'}
        ]

        names = get_cluster_names(labels, docs)

        # Should have names for clusters 0 and 1
        assert 0 in names
        assert 1 in names
        # Noise cluster (-1) should not have a name
        assert -1 not in names
        # Names should be strings
        assert isinstance(names[0], str)
        assert isinstance(names[1], str)

    def test_load_embeddings_missing_file(self):
        """Test error handling for missing embeddings."""
        from cluster_archive import load_embeddings

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_embeddings(Path(tmpdir))


class TestIntegration:
    """Integration tests for the full pipeline."""

    @pytest.mark.slow
    def test_full_pipeline_tinystories(self):
        """Test full ingestion and clustering pipeline with TinyStories."""
        from ingest_training_data import get_tinystories_samples, generate_embeddings, get_sentence_transformer, save_archive
        from cluster_archive import reduce_dimensions, cluster_documents, save_clustered_archive

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Sample 20 docs
            docs = get_tinystories_samples(20)
            assert len(docs) >= 10  # Might get slightly fewer

            # Generate embeddings
            model = get_sentence_transformer()
            embeddings = generate_embeddings(docs, model)
            assert embeddings.shape[0] == len(docs)

            # Reduce dimensions
            coords_2d = reduce_dimensions(embeddings, n_neighbors=5)
            assert coords_2d.shape == (len(docs), 2)

            # Cluster
            labels = cluster_documents(embeddings, min_cluster_size=3)
            assert len(labels) == len(docs)

            # Save
            save_clustered_archive(docs, embeddings, coords_2d, labels, output_dir)

            # Verify output
            assert (output_dir / 'archive_manifest.json').exists()
            assert (output_dir / 'coords_2d.npy').exists()
            assert (output_dir / 'cluster_labels.npy').exists()

            with open(output_dir / 'archive_manifest.json') as f:
                manifest = json.load(f)

            assert manifest['metadata']['totalDocuments'] == len(docs)
            assert 'clusterNames' in manifest


class TestUtils:
    """Utility function tests."""

    def test_embedding_consistency(self):
        """Test that same text produces same embedding."""
        from ingest_training_data import get_sentence_transformer

        model = get_sentence_transformer()
        texts = ["The quick brown fox jumps over the lazy dog."]

        emb1 = model.encode(texts)
        emb2 = model.encode(texts)

        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_cosine_similarity(self):
        """Test semantic similarity in embeddings."""
        from ingest_training_data import get_sentence_transformer

        model = get_sentence_transformer()

        texts = [
            "Cats are furry pets.",
            "Dogs are furry pets.",
            "Quantum mechanics is a physics theory."
        ]

        embeddings = model.encode(texts, normalize_embeddings=True)

        # Similarity between cats and dogs
        sim_cats_dogs = np.dot(embeddings[0], embeddings[1])
        # Similarity between cats and quantum
        sim_cats_quantum = np.dot(embeddings[0], embeddings[2])

        # Cats should be more similar to dogs than to quantum mechanics
        assert sim_cats_dogs > sim_cats_quantum


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
