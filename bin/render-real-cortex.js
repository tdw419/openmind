#!/usr/bin/env node
/**
 * Real Cortex Renderer - Visualizes actual distilgpt2 weights
 *
 * Renders the 79,946 tiles containing real neural weights into
 * a high-resolution image showing the learned patterns.
 *
 * Usage:
 *   node bin/render-real-cortex.js
 */

import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, '..');
const CORTEX_MANIFEST = join(PROJECT_ROOT, 'cortex', 'spatial_manifest.json');
const OUTPUT_DIR = join(PROJECT_ROOT, 'visualizations');

// Color schemes for different layer types
const LAYER_COLORS = {
    'wte': { r: 50, g: 150, b: 255 },      // Token embeddings: Blue
    'wpe': { r: 50, g: 255, b: 200 },       // Position embeddings: Cyan
    'attn_c_attn': { r: 0, g: 255, b: 150 },   // Attention Q,K,V: Green-cyan
    'attn_c_proj': { r: 150, g: 255, b: 0 },   // Attention proj: Yellow-green
    'mlp_c_fc': { r: 255, g: 100, b: 200 },    // MLP up: Pink
    'mlp_c_proj': { r: 200, g: 50, b: 255 },   // MLP down: Purple
    'ln': { r: 255, g: 255, b: 100 },          // Layer norm: Yellow
    'ln_f': { r: 255, g: 200, b: 100 }         // Final LN: Orange
};

function getLayerColor(parameterName) {
    if (parameterName.includes('wte')) return LAYER_COLORS.wte;
    if (parameterName.includes('wpe')) return LAYER_COLORS.wpe;
    if (parameterName.includes('attn_c_attn')) return LAYER_COLORS.attn_c_attn;
    if (parameterName.includes('attn_c_proj')) return LAYER_COLORS.attn_c_proj;
    if (parameterName.includes('mlp_c_fc')) return LAYER_COLORS.mlp_c_fc;
    if (parameterName.includes('mlp_c_proj')) return LAYER_COLORS.mlp_c_proj;
    if (parameterName.includes('ln_f')) return LAYER_COLORS.ln_f;
    if (parameterName.includes('ln')) return LAYER_COLORS.ln;
    return { r: 128, g: 128, b: 128 };
}

function renderRealCortex() {
    console.log('🧠 Rendering Real DistilGPT2 Cortex\n');

    // Load manifest
    const cortex = JSON.parse(readFileSync(CORTEX_MANIFEST, 'utf-8'));
    const tiles = cortex.tiles;
    const metadata = cortex.metadata;

    console.log(`  Model: ${metadata.name}`);
    console.log(`  Parameters: ${metadata.total_parameters?.toLocaleString()}`);
    console.log(`  Tiles: ${tiles.length}`);
    console.log(`  Grid: ${metadata.grid_size}x${metadata.grid_size}\n`);

    // Create image buffer
    // Scale down for manageable image size
    const scale = 8;  // Each tile -> 8x8 pixels
    const width = metadata.grid_size * scale;
    const height = metadata.grid_size * scale;
    const pixels = new Uint8ClampedArray(width * height * 4);
    pixels.fill(5); // Dark background

    console.log('  Rendering tiles...');
    let rendered = 0;
    let errors = 0;

    tiles.forEach(tile => {
        const baseX = tile.x * scale;
        const baseY = tile.y * scale;

        try {
            // Load weight data
            const binPath = join(PROJECT_ROOT, 'cortex', tile.file);
            const data = readFileSync(binPath);
            const floats = new Float32Array(data.buffer);

            // Calculate tile statistics
            let sum = 0, absSum = 0, max = -Infinity, min = Infinity;
            for (let i = 0; i < floats.length; i++) {
                sum += floats[i];
                absSum += Math.abs(floats[i]);
                max = Math.max(max, floats[i]);
                min = Math.min(min, floats[i]);
            }
            const mean = sum / floats.length;
            const absMean = absSum / floats.length;
            const range = max - min;

            // Get base color for layer type
            const baseColor = getLayerColor(tile.parameter);

            // Render tile as scale x scale pixel block
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    const px = baseX + dx;
                    const py = baseY + dy;

                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;

                        // Map weight statistics to color intensity
                        // High absolute mean = bright, low = dim
                        const intensity = Math.min(1, absMean * 5);

                        // Sign determines color shift
                        const signShift = mean > 0 ? 1 : -1;

                        // Apply intensity to base color with sign-based modulation
                        let r = baseColor.r * intensity;
                        let g = baseColor.g * intensity;
                        let b = baseColor.b * intensity;

                        // Sign-based color shift
                        if (signShift > 0) {
                            // Positive weights shift toward cyan/white
                            r = Math.min(255, r * 0.8 + intensity * 50);
                            g = Math.min(255, g + intensity * 30);
                            b = Math.min(255, b + intensity * 30);
                        } else {
                            // Negative weights shift toward purple/red
                            r = Math.min(255, r + intensity * 50);
                            g = Math.min(255, g * 0.8);
                            b = Math.min(255, b * 0.9);
                        }

                        pixels[idx] = Math.floor(r);
                        pixels[idx + 1] = Math.floor(g);
                        pixels[idx + 2] = Math.floor(b);
                        pixels[idx + 3] = 255;
                    }
                }
            }

            rendered++;
            if (rendered % 10000 === 0) {
                console.log(`    ${rendered.toLocaleString()} tiles...`);
            }
        } catch (e) {
            errors++;
        }
    });

    console.log(`\n  Rendered: ${rendered.toLocaleString()} tiles`);
    console.log(`  Errors: ${errors}`);

    // Add legend
    console.log('\n  Adding legend...');
    const legendY = height - 60;

    const legendItems = [
        { name: 'Token Embeddings', color: LAYER_COLORS.wte },
        { name: 'Position Embeddings', color: LAYER_COLORS.wpe },
        { name: 'Attention Q,K,V', color: LAYER_COLORS.attn_c_attn },
        { name: 'Attention Proj', color: LAYER_COLORS.attn_c_proj },
        { name: 'MLP Up', color: LAYER_COLORS.mlp_c_fc },
        { name: 'MLP Down', color: LAYER_COLORS.mlp_c_proj },
        { name: 'Layer Norm', color: LAYER_COLORS.ln }
    ];

    legendItems.forEach((item, i) => {
        const x = 20 + i * 200;
        // Draw color box
        for (let dy = 0; dy < 15; dy++) {
            for (let dx = 0; dx < 15; dx++) {
                const idx = ((legendY + dy) * width + x + dx) * 4;
                pixels[idx] = item.color.r;
                pixels[idx + 1] = item.color.g;
                pixels[idx + 2] = item.color.b;
                pixels[idx + 3] = 255;
            }
        }
        // Draw label (simplified - just a bright line)
        for (let dx = 20; dx < 100; dx++) {
            const idx = ((legendY + 7) * width + x + dx) * 4;
            pixels[idx] = 200;
            pixels[idx + 1] = 200;
            pixels[idx + 2] = 200;
            pixels[idx + 3] = 255;
        }
    });

    // Save as raw then convert to PNG
    const rawPath = join(OUTPUT_DIR, 'real_cortex.raw');
    const pngPath = join(OUTPUT_DIR, 'real_cortex.png');

    writeFileSync(rawPath, pixels);
    execSync(`convert -size ${width}x${height} -depth 8 rgba:${rawPath} ${pngPath}`);

    console.log(`\n✅ Real cortex rendered!`);
    console.log(`   Output: ${pngPath}`);
    console.log(`   Size: ${width}x${height} pixels`);

    // Print layer statistics
    console.log('\n📊 Layer Statistics:');
    Object.entries(cortex.layer_stats || {}).forEach(([name, stats]) => {
        const color = getLayerColor(name);
        const colorName = Object.entries(LAYER_COLORS).find(([k, v]) =>
            v.r === color.r && v.g === color.g && v.b === color.b
        )?.[0] || 'unknown';
        console.log(`   ${name}:`);
        console.log(`     Tiles: ${stats.num_tiles.toLocaleString()}`);
        console.log(`     Mean: ${stats.mean.toFixed(6)}`);
        console.log(`     Std: ${stats.std.toFixed(6)}`);
        console.log(`     Range: [${stats.min.toFixed(4)}, ${stats.max.toFixed(4)}]`);
    });
}

renderRealCortex();
