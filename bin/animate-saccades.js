#!/usr/bin/env node
/**
 * Animated Saccade Renderer - Creates frame-by-frame attention animation
 *
 * Generates individual frames showing saccades animating over time,
 * which can be combined into a GIF or video.
 *
 * Usage:
 *   node bin/animate-saccades.js
 *   node bin/animate-saccades.js --output frames/
 *   convert -delay 50 -loop 0 frames/*.png animation.gif
 */

import { readFileSync, writeFileSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, '..');

const CORTEX_MANIFEST = join(PROJECT_ROOT, 'cortex', 'spatial_manifest.json');
const ARCHIVE_MANIFEST = join(PROJECT_ROOT, 'archive', 'archive_manifest.json');
const ATTENTION_DATA = join(PROJECT_ROOT, 'visualizations', 'real_attention.json');

// Layer colors
const layerColors = [
    [100, 200, 255],  // Layer 0: Cyan
    [100, 255, 200],  // Layer 1: Teal
    [200, 255, 100],  // Layer 2: Yellow-green
    [255, 200, 100],  // Layer 3: Orange
    [255, 100, 200],  // Layer 4: Pink
    [200, 100, 255],  // Layer 5: Purple
];
const embedColor = [50, 150, 255];

function generateFrames() {
    console.log('🎬 Generating Animated Saccade Frames\n');

    // Check for attention data
    if (!existsSync(ATTENTION_DATA)) {
        console.log('❌ No attention data found. Run inference first:');
        console.log('   python bin/inference-engine.py --query "What is gravity?"');
        return;
    }

    const cortex = JSON.parse(readFileSync(CORTEX_MANIFEST, 'utf-8'));
    const archive = JSON.parse(readFileSync(ARCHIVE_MANIFEST, 'utf-8'));
    const attention = JSON.parse(readFileSync(ATTENTION_DATA, 'utf-8'));

    const saccades = attention.saccades || [];
    if (saccades.length === 0) {
        console.log('❌ No saccades in attention data');
        return;
    }

    console.log(`  Query: "${attention.input_text}"`);
    console.log(`  Saccades: ${saccades.length}`);

    // Image dimensions
    const width = 1600;
    const height = 600;

    // Output directory
    const outputDir = join(PROJECT_ROOT, 'visualizations', 'frames');
    mkdirSync(outputDir, { recursive: true });

    // Build tile lookup
    const tileById = new Map();
    cortex.tiles.forEach(tile => tileById.set(tile.id, tile));

    // Group tiles by layer
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

    const layers = ['embed', 0, 1, 2, 3, 4, 5];
    const bandHeight = 500 / layers.length;

    // Cortex/archive offsets
    const cortexOffsetX = 50;
    const cortexOffsetY = 50;
    const archiveOffsetX = 700;
    const archiveOffsetY = 50;

    // Generate frames
    // Each frame shows progressively more saccades animating in
    const totalFrames = Math.min(30, saccades.length);
    const saccadesPerFrame = Math.ceil(saccades.length / totalFrames);

    console.log(`\n  Generating ${totalFrames} frames...`);

    for (let frame = 0; frame < totalFrames; frame++) {
        const pixels = new Uint8ClampedArray(width * height * 4);
        pixels.fill(8); // Dark background

        const saccadesUpTo = Math.min((frame + 1) * saccadesPerFrame, saccades.length);

        // Draw static elements (cortex bands, archive docs)
        drawStaticElements(pixels, width, height, cortex, archive, layers, tilesByLayer,
                          cortexOffsetX, cortexOffsetY, archiveOffsetX, archiveOffsetY, bandHeight);

        // Draw saccades up to this frame with animation
        for (let i = 0; i < saccadesUpTo; i++) {
            const saccade = saccades[i];
            const animProgress = i === saccadesUpTo - 1 ?
                Math.min(1, (frame % 1) + 0.5) : 1; // Latest saccade animates in

            drawSaccade(pixels, width, height, saccade, tileById, tilesByLayer, layers,
                       cortexOffsetX, cortexOffsetY, archiveOffsetX, archiveOffsetY, bandHeight,
                       animProgress, i === saccadesUpTo - 1);
        }

        // Draw frame counter
        drawText(pixels, width, height, `Frame ${frame + 1}/${totalFrames}`, width - 150, 20, [150, 150, 150]);

        // Save frame
        const rawPath = join(outputDir, `frame_${String(frame).padStart(3, '0')}.raw`);
        const pngPath = join(outputDir, `frame_${String(frame).padStart(3, '0')}.png`);
        writeFileSync(rawPath, pixels);
        execSync(`convert -size ${width}x${height} -depth 8 rgba:${rawPath} ${pngPath}`, { stdio: 'pipe' });

        if ((frame + 1) % 10 === 0) {
            console.log(`  Frame ${frame + 1}/${totalFrames}`);
        }
    }

    console.log(`\n✅ Frames generated in: ${outputDir}`);
    console.log(`\n  To create GIF:`);
    console.log(`    convert -delay 5 -loop 0 ${outputDir}/frame_*.png ${PROJECT_ROOT}/visualizations/saccade_animation.gif`);
    console.log(`\n  To create video:`);
    console.log(`    ffmpeg -framerate 10 -i ${outputDir}/frame_%03d.png -c:v libx264 ${PROJECT_ROOT}/visualizations/saccade_animation.mp4`);
}

function drawStaticElements(pixels, width, height, cortex, archive, layers, tilesByLayer,
                           cortexOffsetX, cortexOffsetY, archiveOffsetX, archiveOffsetY, bandHeight) {
    // Draw layer bands
    layers.forEach((layer, bandIdx) => {
        const color = layer === 'embed' ? embedColor : layerColors[layer % layerColors.length];
        const bandY = cortexOffsetY + bandIdx * bandHeight;

        // Layer label
        const label = layer === 'embed' ? 'EMB' : `L${layer}`;
        drawText(pixels, width, height, label, cortexOffsetX - 30, bandY + 5, color);
    });

    // Draw archive documents
    archive.documents.forEach((doc, i) => {
        const dx = archiveOffsetX + (i % 3) * 150;
        const dy = archiveOffsetY + Math.floor(i / 3) * 140;

        // Draw box border
        for (let py = dy; py < dy + 120; py++) {
            for (let px = dx; px < dx + 130; px++) {
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    const idx = (py * width + px) * 4;
                    const isBorder = py < dy + 2 || py >= dy + 118 || px < dx + 2 || px >= dx + 128;
                    if (isBorder) {
                        pixels[idx] = 60;
                        pixels[idx + 1] = 60;
                        pixels[idx + 2] = 60;
                        pixels[idx + 3] = 255;
                    }
                }
            }
        }
    });
}

function drawSaccade(pixels, width, height, saccade, tileById, tilesByLayer, layers,
                    cortexOffsetX, cortexOffsetY, archiveOffsetX, archiveOffsetY, bandHeight,
                    progress, isLatest) {
    const tile = tileById.get(saccade.tile_id);
    if (!tile) return;

    const layerMatch = tile.parameter?.match(/layer_(\d+)/);
    const layer = layerMatch ? parseInt(layerMatch[1]) : 'embed';
    const layerIdx = layers.indexOf(layer);
    if (layerIdx === -1) return;

    const layerTiles = tilesByLayer[layer] || [];
    const tileIdx = layerTiles.findIndex(t => t.id === tile.id);

    // Source position
    const sx = cortexOffsetX + (tileIdx % 50) * 10 + 3;
    const sy = cortexOffsetY + layerIdx * bandHeight + 20 + Math.floor(tileIdx / 50) * 10 + 3;

    // Target position
    const tx = archiveOffsetX + (saccade.doc_id % 3) * 150 + 65;
    const ty = archiveOffsetY + Math.floor(saccade.doc_id / 3) * 140 + 60;

    // Line color based on layer
    const layerColor = layer === 'embed' ? embedColor : layerColors[layer % layerColors.length];
    const semSim = saccade.semantic_similarity || 0;
    const brightness = isLatest ? 1.0 : (0.3 + semSim * 0.4);

    // Draw animated line
    const dx = tx - sx;
    const dy = ty - sy;
    const totalSteps = Math.max(Math.abs(dx), Math.abs(dy));
    const steps = Math.floor(totalSteps * progress);

    for (let i = 0; i <= steps; i++) {
        const t = i / totalSteps;
        const x = Math.floor(sx + dx * t);
        const y = Math.floor(sy + dy * t);

        if (x >= 0 && x < width && y >= 0 && y < height) {
            const idx = (y * width + x) * 4;
            const fade = 1 - Math.abs(t - 0.5) * 0.3;

            // Glow effect for latest saccade
            const glowSize = isLatest ? 2 : 1;
            for (let gy = -glowSize; gy <= glowSize; gy++) {
                for (let gx = -glowSize; gx <= glowSize; gx++) {
                    const px = x + gx;
                    const py = y + gy;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const pidx = (py * width + px) * 4;
                        const dist = Math.sqrt(gx * gx + gy * gy);
                        const glowFade = Math.max(0, 1 - dist / (glowSize + 1));
                        pixels[pidx] = Math.floor(layerColor[0] * brightness * fade * glowFade);
                        pixels[pidx + 1] = Math.floor(layerColor[1] * brightness * fade * glowFade);
                        pixels[pidx + 2] = Math.floor(layerColor[2] * brightness * fade * glowFade);
                        pixels[pidx + 3] = Math.floor(200 * brightness * glowFade);
                    }
                }
            }
        }
    }
}

function drawText(pixels, width, height, text, startX, startY, color) {
    for (let i = 0; i < text.length; i++) {
        for (let dy = 0; dy < 10; dy++) {
            for (let dx = 0; dx < 6; dx++) {
                if ((dx + dy + text.charCodeAt(i)) % 3 !== 0) {
                    const px = startX + i * 7 + dx;
                    const py = startY + dy;
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
}

generateFrames();
