#!/usr/bin/env node
/**
 * Real Saccade Renderer - Visualizes actual neural attention patterns
 *
 * Renders real attention from distilgpt2 inference over the actual cortex
 * tiles and archive documents.
 *
 * Usage:
 *   python3 bin/inference-engine.py --query "What is gravity?"
 *   node bin/render-real-saccades.js
 */

import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { execSync } from 'child_process';

import { dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, '..');
const CORTEX_MANIFEST = join(PROJECT_ROOT, 'cortex', 'spatial_manifest.json');
const ARCHIVE_MANIFEST = join(PROJECT_ROOT, 'archive', 'archive_manifest.json');
const ATTENTION_DATA = join(PROJECT_ROOT, 'visualizations', 'real_attention.json');
const OUTPUT_DIR = join(PROJECT_ROOT, 'visualizations');

function renderRealSaccades() {
    console.log('⚡ Rendering Real Neural Saccades\n');

    // Load data
    const cortex = JSON.parse(readFileSync(CORTEX_MANIFEST, 'utf-8'));
    const archive = JSON.parse(readFileSync(ARCHIVE_MANIFEST, 'utf-8'));
    const attention = JSON.parse(readFileSync(ATTENTION_DATA, 'utf-8'));

    console.log(`  Query: "${attention.input_text}"`);
    console.log(`  Tokens: ${attention.num_tokens}`);
    console.log(`  Layers: ${attention.num_layers}`);
    console.log(`  Saccades: ${attention.saccades.length}\n`);

    // Image dimensions
    const width = 1600;
    const height = 600;
    const pixels = new Uint8ClampedArray(width * height * 4);
    pixels.fill(8); // Dark background

    // Cortex region (left side) - scaled to fit
    const cortexScale = 1.5;
    const cortexOffsetX = 50;
    const cortexOffsetY = 50;
    const cortexWidth = 500;
    const cortexHeight = 500;

    // Archive region (right side)
    const archiveOffsetX = 700;
    const archiveOffsetY = 50;

    // Build tile lookup by ID
    const tileById = new Map();
    cortex.tiles.forEach(tile => tileById.set(tile.id, tile));

    // Group tiles by layer for visualization
    const tilesByLayer = {};
    cortex.tiles.forEach(tile => {
        const layerMatch = tile.parameter?.match(/layer_(\d+)/);
        if (layerMatch) {
            const layer = parseInt(layerMatch[1]);
            if (!tilesByLayer[layer]) tilesByLayer[layer] = [];
            tilesByLayer[layer].push(tile);
        } else if (tile.parameter?.includes('wte')) {
            if (!tilesByLayer['embed']) tilesByLayer['embed'] = [];
            tilesByLayer['embed'].push(tile);
        }
    });

    // Layer colors
    const layerColors = [
        [100, 200, 255],  // Layer 0: Cyan
        [100, 255, 200],  // Layer 1: Teal
        [200, 255, 100],  // Layer 2: Yellow-green
        [255, 200, 100],  // Layer 3: Orange
        [255, 100, 200],  // Layer 4: Pink
        [200, 100, 255],  // Layer 5: Purple
    ];
    const embedColor = [50, 150, 255]; // Embeddings: Blue

    // Count active tiles from saccades
    const activeTiles = new Map();
    attention.saccades.forEach(s => {
        const count = activeTiles.get(s.tile_id) || { count: 0, totalIntensity: 0 };
        count.count++;
        count.totalIntensity += s.intensity;
        activeTiles.set(s.tile_id, count);
    });

    console.log(`  Active tiles: ${activeTiles.size}`);

    // Draw cortex - simplified view showing layer bands
    const layers = ['embed', 0, 1, 2, 3, 4, 5];
    const bandHeight = cortexHeight / layers.length;

    layers.forEach((layer, bandIdx) => {
        const tiles = tilesByLayer[layer] || [];
        const color = layer === 'embed' ? embedColor : layerColors[layer % layerColors.length];
        const bandY = cortexOffsetY + bandIdx * bandHeight;

        // Draw layer label
        const label = layer === 'embed' ? 'EMB' : `L${layer}`;
        for (let i = 0; i < label.length; i++) {
            for (let dy = 0; dy < 10; dy++) {
                for (let dx = 0; dx < 6; dx++) {
                    if ((dx + dy + label.charCodeAt(i)) % 3 !== 0) {
                        const px = cortexOffsetX - 30 + i * 7 + dx;
                        const py = bandY + 5 + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            const idx = (py * width + px) * 4;
                            pixels[idx] = color[0];
                            pixels[idx + 1] = color[1];
                            pixels[idx + 2] = color[2];
                            pixels[idx + 3] = 255;
                        }
                    }
                }
            }
        }

        // Draw tiles as dots (sample for visibility)
        const sampleStep = Math.max(1, Math.floor(tiles.length / 200));
        tiles.forEach((tile, i) => {
            if (i % sampleStep !== 0) return;

            const tx = cortexOffsetX + (i % 50) * 10;
            const ty = bandY + 20 + Math.floor(i / 50) * 10;

            // Check if this tile is active
            const activity = activeTiles.get(tile.id);
            const isActive = !!activity;
            const avgIntensity = isActive ? activity.totalIntensity / activity.count : 0;

            // Draw tile dot
            const dotSize = isActive ? 6 : 3;
            for (let dy = 0; dy < dotSize; dy++) {
                for (let dx = 0; dx < dotSize; dx++) {
                    const px = tx + dx;
                    const py = ty + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        if (isActive) {
                            // Active tiles glow bright
                            const brightness = 0.5 + avgIntensity;
                            pixels[idx] = Math.min(255, Math.floor(color[0] * brightness + 100));
                            pixels[idx + 1] = Math.min(255, Math.floor(color[1] * brightness + 100));
                            pixels[idx + 2] = Math.min(255, Math.floor(color[2] * brightness + 100));
                        } else {
                            // Inactive tiles are dim
                            pixels[idx] = Math.floor(color[0] * 0.2);
                            pixels[idx + 1] = Math.floor(color[1] * 0.2);
                            pixels[idx + 2] = Math.floor(color[2] * 0.2);
                        }
                        pixels[idx + 3] = 255;
                    }
                }
            }
        });
    });

    // Draw archive documents
    const docColors = {
        physics: [100, 200, 255],
        math: [255, 200, 100],
        biology: [100, 255, 150],
        history: [255, 180, 200],
        language: [200, 150, 255],
        programming: [255, 255, 100]
    };

    // Count attention per document and track max similarity
    const docAttention = new Map();
    const docMaxSim = new Map();
    attention.saccades.forEach(s => {
        const count = docAttention.get(s.doc_id) || 0;
        docAttention.set(s.doc_id, count + 1);
        const prevSim = docMaxSim.get(s.doc_id) || 0;
        docMaxSim.set(s.doc_id, Math.max(prevSim, s.semantic_similarity || 0));
    });

    archive.documents.forEach((doc, i) => {
        const dx = archiveOffsetX + (i % 3) * 150;
        const dy = archiveOffsetY + Math.floor(i / 3) * 140;
        const color = docColors[doc.category] || [128, 128, 128];

        const isActive = docAttention.has(i);
        const attentionCount = docAttention.get(i) || 0;

        // Draw document box
        for (let py = dy; py < dy + 120; py++) {
            for (let px = dx; px < dx + 130; px++) {
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    const idx = (py * width + px) * 4;
                    const isBorder = py < dy + 3 || py >= dy + 117 || px < dx + 3 || px >= dx + 127;

                    const maxSim = docMaxSim.get(i) || 0;
                    if (isBorder) {
                        // Bright white border for high-similarity targets
                        if (isActive && maxSim > 0.3) {
                            pixels[idx] = 255;
                            pixels[idx + 1] = 255;
                            pixels[idx + 2] = 255;
                        } else {
                            pixels[idx] = color[0];
                            pixels[idx + 1] = color[1];
                            pixels[idx + 2] = color[2];
                        }
                    } else if (isActive) {
                        // Glow intensity scales with semantic similarity
                        const glow = Math.floor(maxSim * 200);
                        pixels[idx] = Math.min(255, Math.floor(color[0] * 0.3) + glow);
                        pixels[idx + 1] = Math.min(255, Math.floor(color[1] * 0.3) + glow);
                        pixels[idx + 2] = Math.min(255, Math.floor(color[2] * 0.3) + glow);
                    } else {
                        // Dim inactive docs
                        pixels[idx] = Math.floor(color[0] * 0.08);
                        pixels[idx + 1] = Math.floor(color[1] * 0.08);
                        pixels[idx + 2] = Math.floor(color[2] * 0.08);
                    }
                    pixels[idx + 3] = 255;
                }
            }
        }

        // Draw category label
        const label = doc.category.slice(0, 8).toUpperCase();
        for (let li = 0; li < label.length; li++) {
            const lx = dx + 8 + li * 8;
            const ly = dy + 8;
            for (let ly2 = 0; ly2 < 10; ly2++) {
                for (let lx2 = 0; lx2 < 6; lx2++) {
                    if ((lx2 + ly2 + label.charCodeAt(li)) % 2 === 0) {
                        const px = lx + lx2;
                        const py = ly + ly2;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            const idx = (py * width + px) * 4;
                            pixels[idx] = color[0];
                            pixels[idx + 1] = color[1];
                            pixels[idx + 2] = color[2];
                            pixels[idx + 3] = 255;
                        }
                    }
                }
            }
        }

        // Show attention count if active
        if (isActive) {
            const countStr = attentionCount.toString();
            for (let ci = 0; ci < countStr.length; ci++) {
                for (let dy2 = 0; dy2 < 8; dy2++) {
                    for (let dx2 = 0; dx2 < 5; dx2++) {
                        if ((dx2 + dy2 + countStr.charCodeAt(ci)) % 2 === 0) {
                            const px = dx + 100 + ci * 6 + dx2;
                            const py = dy + 100 + dy2;
                            if (px >= 0 && px < width && py >= 0 && py < height) {
                                const idx = (py * width + px) * 4;
                                pixels[idx] = 255;
                                pixels[idx + 1] = 255;
                                pixels[idx + 2] = 255;
                                pixels[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
    });

    // Draw saccade lines (attention connections)
    console.log('  Drawing attention connections...');
    let drawnSaccades = 0;

    // Sample saccades to avoid overdraw
    const sampleRate = Math.max(1, Math.floor(attention.saccades.length / 50));
    attention.saccades.forEach((saccade, saccadeIdx) => {
        if (saccadeIdx % sampleRate !== 0) return;

        const tile = tileById.get(saccade.tile_id);
        if (!tile) return;

        // Find layer for this tile
        const layerMatch = tile.parameter?.match(/layer_(\d+)/);
        const layer = layerMatch ? parseInt(layerMatch[1]) : 'embed';
        const layerIdx = layers.indexOf(layer);
        if (layerIdx === -1) return;

        // Get layer tiles to find position within band
        const layerTiles = tilesByLayer[layer] || [];
        const tileIdx = layerTiles.findIndex(t => t.id === tile.id);

        // Calculate source position
        const sx = cortexOffsetX + (tileIdx % 50) * 10 + 3;
        const sy = cortexOffsetY + layerIdx * bandHeight + 20 + Math.floor(tileIdx / 50) * 10 + 3;

        // Calculate target position (archive doc)
        const tx = archiveOffsetX + (saccade.doc_id % 3) * 150 + 65;
        const ty = archiveOffsetY + Math.floor(saccade.doc_id / 3) * 140 + 60;

        // Draw line with Bresenham's algorithm
        const dx = tx - sx;
        const dy = ty - sy;
        const steps = Math.max(Math.abs(dx), Math.abs(dy));
        const attnIntensity = Math.min(1, saccade.intensity * 2);
        const semSim = saccade.semantic_similarity || 0;
        // Combined score: attention weight × semantic relevance
        const intensity = Math.min(1, attnIntensity * (0.3 + semSim * 0.7));

        // Color by layer with semantic similarity affecting brightness
        const layerColor = layer === 'embed' ? embedColor : layerColors[layer % layerColors.length];
        const brightness = 0.4 + semSim * 0.6; // Semantic similarity affects brightness
        const r = Math.floor(layerColor[0] * brightness);
        const g = Math.floor(layerColor[1] * brightness);
        const b = Math.floor(layerColor[2] * brightness);

        for (let i = 0; i <= steps; i++) {
            const t = i / steps;
            const x = Math.floor(sx + dx * t);
            const y = Math.floor(sy + dy * t);

            if (x >= 0 && x < width && y >= 0 && y < height) {
                const idx = (y * width + x) * 4;
                const fade = 1 - Math.abs(t - 0.5) * 0.5;
                pixels[idx] = Math.floor(r * intensity * fade);
                pixels[idx + 1] = Math.floor(g * intensity * fade);
                pixels[idx + 2] = Math.floor(b * intensity * fade);
                pixels[idx + 3] = Math.floor(220 * intensity);
            }
        }
        drawnSaccades++;
    });

    console.log(`  Drew ${drawnSaccades} saccade lines`);

    // Add title
    const title = `ATTENTION: "${attention.input_text.slice(0, 35)}..."`;
    for (let i = 0; i < title.length; i++) {
        const tx = 50 + i * 8;
        for (let dy = 0; dy < 12; dy++) {
            for (let dx = 0; dx < 6; dx++) {
                if ((dx + dy + title.charCodeAt(i)) % 2 === 0) {
                    const px = tx + dx;
                    const py = height - 25 + dy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        pixels[idx] = 200;
                        pixels[idx + 1] = 200;
                        pixels[idx + 2] = 200;
                        pixels[idx + 3] = 255;
                    }
                }
            }
        }
    }

    // Draw layer legend (bottom right)
    const legendX = width - 120;
    const legendY = height - 100;
    const legendLayers = [
        { name: 'EMB', color: embedColor },
        { name: 'L0', color: layerColors[0] },
        { name: 'L1', color: layerColors[1] },
        { name: 'L2', color: layerColors[2] },
        { name: 'L3', color: layerColors[3] },
        { name: 'L4', color: layerColors[4] },
        { name: 'L5', color: layerColors[5] }
    ];

    legendLayers.forEach((item, i) => {
        const ly = legendY + i * 12;
        // Color swatch
        for (let dy = 0; dy < 8; dy++) {
            for (let dx = 0; dx < 8; dx++) {
                const px = legendX + dx;
                const py = ly + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    const idx = (py * width + px) * 4;
                    pixels[idx] = item.color[0];
                    pixels[idx + 1] = item.color[1];
                    pixels[idx + 2] = item.color[2];
                    pixels[idx + 3] = 255;
                }
            }
        }
        // Label
        for (let li = 0; li < item.name.length; li++) {
            for (let dy = 0; dy < 8; dy++) {
                for (let dx = 0; dx < 5; dx++) {
                    if ((dx + dy + item.name.charCodeAt(li)) % 2 === 0) {
                        const px = legendX + 12 + li * 6 + dx;
                        const py = ly + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            const idx = (py * width + px) * 4;
                            pixels[idx] = 180;
                            pixels[idx + 1] = 180;
                            pixels[idx + 2] = 180;
                            pixels[idx + 3] = 255;
                        }
                    }
                }
            }
        }
    });

    // Save as raw then convert to PNG
    const rawPath = join(OUTPUT_DIR, 'real_saccades.raw');
    const pngPath = join(OUTPUT_DIR, 'real_saccades.png');

    writeFileSync(rawPath, pixels);
    execSync(`convert -size ${width}x${height} -depth 8 rgba:${rawPath} ${pngPath}`);

    console.log(`\n✅ Real saccades rendered!`);
    console.log(`   Output: ${pngPath}`);
    console.log(`\n   Active tiles: ${activeTiles.size}`);
    console.log(`   Active docs: ${docAttention.size}`);

    // Print attention distribution
    console.log('\n📊 Attention by Document:');
    const sortedDocs = [...docAttention.entries()].sort((a, b) => b[1] - a[1]);
    sortedDocs.slice(0, 5).forEach(([docId, count]) => {
        const doc = archive.documents[docId];
        if (doc) {
            console.log(`   Doc ${docId} [${doc.category}]: ${count} connections`);
            console.log(`     "${doc.text.slice(0, 50)}..."`);
        }
    });
}

renderRealSaccades();
