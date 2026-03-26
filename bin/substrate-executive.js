#!/usr/bin/env node
/**
 * Substrate Executive Protocol - Glass Box LLM with Map Agency
 *
 * This gives the Glass Box LLM the ability to actually manipulate
 * the Infinite Map. It runs inference, extracts intent, and executes
 * map operations while visualizing the decision process.
 *
 * Usage:
 *   node bin/substrate-executive.js --query "Organize the physics documents together"
 *   node bin/substrate-executive.js --query "Move biology content to the north"
 *   node bin/substrate-executive.js --auto  # Let the AI self-organize
 */

import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = join(__dirname, '..');

const CORTEX_MANIFEST = join(PROJECT_ROOT, 'cortex', 'spatial_manifest.json');
const ARCHIVE_MANIFEST = join(PROJECT_ROOT, 'archive', 'archive_manifest.json');
const MAP_STATE = join(PROJECT_ROOT, 'map_state.json');
const OUTPUT_DIR = join(PROJECT_ROOT, 'visualizations');

// Map operation types
const OPERATIONS = {
    MOVE: 'MOVE',
    ALLOCATE: 'ALLOCATE',
    CLUSTER: 'CLUSTER',
    EXPAND: 'EXPAND',
    COMPRESS: 'COMPRESS',
    NOOP: 'NOOP'
};

// Intent keywords for operation detection
const INTENT_PATTERNS = {
    [OPERATIONS.MOVE]: ['move', 'shift', 'relocate', 'slide', 'push', 'pull'],
    [OPERATIONS.CLUSTER]: ['organize', 'cluster', 'group', 'together', 'gather', 'collect'],
    [OPERATIONS.EXPAND]: ['expand', 'grow', 'spread', 'enlarge', 'stretch'],
    [OPERATIONS.COMPRESS]: ['compress', 'shrink', 'compact', 'condense', 'squeeze'],
    [OPERATIONS.ALLOCATE]: ['allocate', 'create', 'add', 'new', 'spawn', 'generate']
};

// Direction vectors
const DIRECTIONS = {
    north: { dx: 0, dy: -100 },
    south: { dx: 0, dy: 100 },
    east: { dx: 100, dy: 0 },
    west: { dx: -100, dy: 0 },
    center: { dx: 0, dy: 0 },
    'top-left': { dx: -70, dy: -70 },
    'top-right': { dx: 70, dy: -70 },
    'bottom-left': { dx: -70, dy: 70 },
    'bottom-right': { dx: 70, dy: 70 }
};

// Category colors for visualization
const CATEGORY_COLORS = {
    physics: [100, 200, 255],
    math: [255, 200, 100],
    biology: [100, 255, 150],
    history: [255, 180, 200],
    language: [200, 150, 255],
    programming: [255, 255, 100]
};

class SubstrateExecutive {
    constructor() {
        this.cortex = this.loadManifest(CORTEX_MANIFEST);
        this.archive = this.loadManifest(ARCHIVE_MANIFEST);
        this.mapState = this.loadMapState();
        this.operations = [];
        this.saccades = [];
    }

    loadManifest(path) {
        try {
            return JSON.parse(readFileSync(path, 'utf-8'));
        } catch (e) {
            console.warn(`Warning: Could not load ${path}`);
            return null;
        }
    }

    loadMapState() {
        try {
            return JSON.parse(readFileSync(MAP_STATE, 'utf-8'));
        } catch (e) {
            // Initialize default state
            return {
                version: 1,
                lastModified: new Date().toISOString(),
                sectors: {},
                pendingOperations: [],
                history: []
            };
        }
    }

    saveMapState() {
        this.mapState.lastModified = new Date().toISOString();
        writeFileSync(MAP_STATE, JSON.stringify(this.mapState, null, 2));
    }

    /**
     * Detect operation type from query text
     */
    detectIntent(query) {
        const lowerQuery = query.toLowerCase();

        // Check for each operation pattern
        for (const [op, patterns] of Object.entries(INTENT_PATTERNS)) {
            for (const pattern of patterns) {
                if (lowerQuery.includes(pattern)) {
                    return op;
                }
            }
        }

        return OPERATIONS.NOOP;
    }

    /**
     * Detect target category from query
     */
    detectCategory(query) {
        const lowerQuery = query.toLowerCase();
        const categories = Object.keys(CATEGORY_COLORS);

        for (const cat of categories) {
            if (lowerQuery.includes(cat)) {
                return cat;
            }
        }

        return null; // All categories
    }

    /**
     * Detect direction from query
     */
    detectDirection(query) {
        const lowerQuery = query.toLowerCase();

        for (const [dir, vector] of Object.entries(DIRECTIONS)) {
            if (lowerQuery.includes(dir)) {
                return { direction: dir, ...vector };
            }
        }

        return { direction: 'center', dx: 0, dy: 0 };
    }

    /**
     * Execute a MOVE operation
     */
    executeMove(category, direction) {
        const docs = this.archive?.documents || [];
        const operations = [];

        docs.forEach((doc, i) => {
            if (category === null || doc.category === category) {
                const oldCoords = { ...doc.coords };
                const newCoords = {
                    x: doc.coords.x + direction.dx,
                    y: doc.coords.y + direction.dy
                };

                operations.push({
                    type: OPERATIONS.MOVE,
                    target: `doc_${i}`,
                    category: doc.category,
                    from: oldCoords,
                    to: newCoords,
                    reason: `Moving ${doc.category} content ${direction.direction}`
                });

                // Update the document coordinates
                doc.coords = newCoords;
            }
        });

        return operations;
    }

    /**
     * Execute a CLUSTER operation - group by category
     */
    executeCluster(category) {
        const docs = this.archive?.documents || [];
        const operations = [];

        // Group documents by category
        const byCategory = {};
        docs.forEach((doc, i) => {
            if (!byCategory[doc.category]) {
                byCategory[doc.category] = [];
            }
            byCategory[doc.category].push({ doc, index: i });
        });

        // Calculate cluster centers (spiral outward from origin)
        const categories = Object.keys(byCategory);
        const clusterRadius = 200;
        const baseX = 5000;
        const baseY = 1000;

        categories.forEach((cat, catIdx) => {
            if (category !== null && cat !== category) return;

            const angle = (catIdx / categories.length) * 2 * Math.PI;
            const centerX = baseX + Math.cos(angle) * clusterRadius * 2;
            const centerY = baseY + Math.sin(angle) * clusterRadius * 2;

            byCategory[cat].forEach(({ doc, index }, i) => {
                const oldCoords = { ...doc.coords };

                // Position within cluster
                const subAngle = (i / byCategory[cat].length) * 2 * Math.PI;
                const subRadius = 50 + i * 30;
                const newCoords = {
                    x: Math.floor(centerX + Math.cos(subAngle) * subRadius),
                    y: Math.floor(centerY + Math.sin(subAngle) * subRadius)
                };

                operations.push({
                    type: OPERATIONS.CLUSTER,
                    target: `doc_${index}`,
                    category: cat,
                    from: oldCoords,
                    to: newCoords,
                    reason: `Clustering ${cat} documents together`
                });

                doc.coords = newCoords;
            });
        });

        return operations;
    }

    /**
     * Execute an EXPAND operation - spread content outward
     */
    executeExpand(category) {
        const docs = this.archive?.documents || [];
        const operations = [];

        // Calculate centroid
        let cx = 0, cy = 0, count = 0;
        docs.forEach(doc => {
            if (category === null || doc.category === category) {
                cx += doc.coords.x;
                cy += doc.coords.y;
                count++;
            }
        });
        if (count > 0) {
            cx /= count;
            cy /= count;
        }

        // Push outward from centroid
        docs.forEach((doc, i) => {
            if (category === null || doc.category === category) {
                const dx = doc.coords.x - cx;
                const dy = doc.coords.y - cy;
                const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                const scale = 1.5;

                const oldCoords = { ...doc.coords };
                const newCoords = {
                    x: Math.floor(cx + (dx / dist) * dist * scale),
                    y: Math.floor(cy + (dy / dist) * dist * scale)
                };

                operations.push({
                    type: OPERATIONS.EXPAND,
                    target: `doc_${i}`,
                    category: doc.category,
                    from: oldCoords,
                    to: newCoords,
                    reason: `Expanding ${doc.category} sector`
                });

                doc.coords = newCoords;
            }
        });

        return operations;
    }

    /**
     * Run inference to get attention-based context
     */
    async runInference(query) {
        return new Promise((resolve, reject) => {
            try {
                // Run the Python inference engine
                const result = execSync(
                    `python3 ${join(PROJECT_ROOT, 'bin', 'inference-engine.py')} --query "${query}"`,
                    { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 }
                );

                // Load the attention data
                const attentionPath = join(PROJECT_ROOT, 'visualizations', 'real_attention.json');
                const attention = JSON.parse(readFileSync(attentionPath, 'utf-8'));
                resolve(attention);
            } catch (e) {
                console.warn('Inference failed, using simulated attention:', e.message);
                resolve(null);
            }
        });
    }

    /**
     * Process a query and execute operations
     */
    async process(query) {
        console.log(`\n🎯 Substrate Executive Processing: "${query}"\n`);

        // Detect intent
        const operation = this.detectIntent(query);
        const category = this.detectCategory(query);
        const direction = this.detectDirection(query);

        console.log(`  Detected operation: ${operation}`);
        console.log(`  Target category: ${category || 'all'}`);
        console.log(`  Direction: ${direction.direction}\n`);

        // Run inference for attention context
        console.log('  Running neural inference...');
        const attention = await this.runInference(query);

        if (attention) {
            console.log(`  Attention: ${attention.saccades?.length || 0} connections`);
            this.saccades = attention.saccades || [];
        }

        // Execute operation
        let operations = [];

        switch (operation) {
            case OPERATIONS.MOVE:
                operations = this.executeMove(category, direction);
                break;
            case OPERATIONS.CLUSTER:
                operations = this.executeCluster(category);
                break;
            case OPERATIONS.EXPAND:
                operations = this.executeExpand(category);
                break;
            case OPERATIONS.NOOP:
            default:
                console.log('  No map operation detected. Analyzing attention only.');
                break;
        }

        this.operations = operations;

        // Save updated state
        if (operations.length > 0) {
            this.mapState.history.push({
                timestamp: new Date().toISOString(),
                query,
                operation,
                operationsCount: operations.length
            });
            this.saveMapState();

            // Update archive manifest with new coordinates
            writeFileSync(ARCHIVE_MANIFEST, JSON.stringify(this.archive, null, 2));
            console.log(`\n  ✅ Executed ${operations.length} operations`);
        }

        // Generate visualization
        await this.renderDecisionMap(query, operation);

        return {
            operation,
            category,
            direction: direction.direction,
            operationsExecuted: operations.length,
            attentionConnections: this.saccades.length
        };
    }

    /**
     * Render the decision map showing operations and attention
     */
    async renderDecisionMap(query, operation) {
        console.log('\n  🗺️  Rendering decision map...');

        const width = 1600;
        const height = 600;
        const pixels = new Uint8ClampedArray(width * height * 4);
        pixels.fill(8);

        // Regions
        const cortexX = 50;
        const cortexY = 50;
        const archiveX = 700;
        const archiveY = 50;

        // Draw layer bands (simplified cortex)
        const layers = ['EMB', 'L0', 'L1', 'L2', 'L3', 'L4', 'L5'];
        const bandHeight = 70;
        const layerColors = [
            [50, 150, 255],   // EMB: Blue
            [100, 200, 255],  // L0: Cyan
            [100, 255, 200],  // L1: Teal
            [200, 255, 100],  // L2: Yellow-green
            [255, 200, 100],  // L3: Orange
            [255, 100, 200],  // L4: Pink
            [200, 100, 255]   // L5: Purple
        ];

        layers.forEach((label, i) => {
            const y = cortexY + i * bandHeight;
            const color = layerColors[i];

            // Draw band background
            for (let py = y; py < y + bandHeight - 5; py++) {
                for (let px = cortexX; px < cortexX + 600; px++) {
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        pixels[idx] = Math.floor(color[0] * 0.15);
                        pixels[idx + 1] = Math.floor(color[1] * 0.15);
                        pixels[idx + 2] = Math.floor(color[2] * 0.15);
                        pixels[idx + 3] = 255;
                    }
                }
            }

            // Draw label
            for (let li = 0; li < label.length; li++) {
                for (let dy = 0; dy < 10; dy++) {
                    for (let dx = 0; dx < 6; dx++) {
                        if ((dx + dy + label.charCodeAt(li)) % 3 !== 0) {
                            const px = cortexX - 30 + li * 7 + dx;
                            const py = y + 10 + dy;
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
        });

        // Draw archive documents with operation indicators
        const docs = this.archive?.documents || [];

        docs.forEach((doc, i) => {
            const dx = archiveX + (i % 3) * 150;
            const dy = archiveY + Math.floor(i / 3) * 140;
            const color = CATEGORY_COLORS[doc.category] || [128, 128, 128];

            // Find operation for this doc
            const docOp = this.operations.find(op => op.target === `doc_${i}`);
            const isTarget = !!docOp;

            // Draw document box
            for (let py = dy; py < dy + 120; py++) {
                for (let px = dx; px < dx + 130; px++) {
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        const isBorder = py < dy + 3 || py >= dy + 117 || px < dx + 3 || px >= dx + 127;

                        if (isBorder) {
                            // Highlight border for targeted docs
                            if (isTarget) {
                                pixels[idx] = 255;
                                pixels[idx + 1] = 255;
                                pixels[idx + 2] = 100;
                            } else {
                                pixels[idx] = color[0];
                                pixels[idx + 1] = color[1];
                                pixels[idx + 2] = color[2];
                            }
                        } else if (isTarget) {
                            // Glow for targeted docs
                            pixels[idx] = Math.min(255, color[0] + 50);
                            pixels[idx + 1] = Math.min(255, color[1] + 50);
                            pixels[idx + 2] = Math.min(255, color[2] + 50);
                        } else {
                            pixels[idx] = Math.floor(color[0] * 0.2);
                            pixels[idx + 1] = Math.floor(color[1] * 0.2);
                            pixels[idx + 2] = Math.floor(color[2] * 0.2);
                        }
                        pixels[idx + 3] = 255;
                    }
                }
            }

            // Draw category label
            const label = doc.category.slice(0, 8).toUpperCase();
            for (let li = 0; li < label.length; li++) {
                for (let dy2 = 0; dy2 < 10; dy2++) {
                    for (let dx2 = 0; dx2 < 6; dx2++) {
                        if ((dx2 + dy2 + label.charCodeAt(li)) % 2 === 0) {
                            const px = dx + 8 + li * 8 + dx2;
                            const py = dy + 8 + dy2;
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

            // Draw operation indicator arrow
            if (docOp) {
                const arrowX = dx + 115;
                const arrowY = dy + 60;

                // Direction arrow
                const moveDx = docOp.to.x > docOp.from.x ? 1 : (docOp.to.x < docOp.from.x ? -1 : 0);
                const moveDy = docOp.to.y > docOp.from.y ? 1 : (docOp.to.y < docOp.from.y ? -1 : 0);

                for (let ai = 0; ai < 10; ai++) {
                    const px = arrowX + moveDx * ai;
                    const py = arrowY + moveDy * ai;
                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        pixels[idx] = 255;
                        pixels[idx + 1] = 255;
                        pixels[idx + 2] = 0;
                        pixels[idx + 3] = 255;
                    }
                }
            }
        });

        // Draw saccade connections
        if (this.saccades.length > 0) {
            const sampleRate = Math.max(1, Math.floor(this.saccades.length / 30));
            this.saccades.forEach((saccade, i) => {
                if (i % sampleRate !== 0) return;

                const sx = cortexX + 300;
                const sy = cortexY + 250 + (saccade.layer || 0) * 30;
                const tx = archiveX + (saccade.doc_id % 3) * 150 + 65;
                const ty = archiveY + Math.floor(saccade.doc_id / 3) * 140 + 60;

                const dx = tx - sx;
                const dy = ty - sy;
                const steps = Math.max(Math.abs(dx), Math.abs(dy));

                for (let si = 0; si <= steps; si++) {
                    const t = si / steps;
                    const px = Math.floor(sx + dx * t);
                    const py = Math.floor(sy + dy * t);

                    if (px >= 0 && px < width && py >= 0 && py < height) {
                        const idx = (py * width + px) * 4;
                        const brightness = Math.floor(200 * saccade.intensity);
                        pixels[idx] = brightness;
                        pixels[idx + 1] = brightness;
                        pixels[idx + 2] = Math.floor(brightness * 0.3);
                        pixels[idx + 3] = 150;
                    }
                }
            });
        }

        // Add operation type label
        const opLabel = `OPERATION: ${operation}`;
        for (let i = 0; i < opLabel.length; i++) {
            for (let dy = 0; dy < 12; dy++) {
                for (let dx = 0; dx < 6; dx++) {
                    if ((dx + dy + opLabel.charCodeAt(i)) % 2 === 0) {
                        const px = 50 + i * 8 + dx;
                        const py = height - 45 + dy;
                        if (px >= 0 && px < width && py >= 0 && py < height) {
                            const idx = (py * width + px) * 4;
                            pixels[idx] = operation === OPERATIONS.NOOP ? 150 : 255;
                            pixels[idx + 1] = 255;
                            pixels[idx + 2] = operation === OPERATIONS.NOOP ? 150 : 100;
                            pixels[idx + 3] = 255;
                        }
                    }
                }
            }
        }

        // Add query label
        const queryLabel = `QUERY: "${query.slice(0, 40)}"`;
        for (let i = 0; i < queryLabel.length; i++) {
            for (let dy = 0; dy < 12; dy++) {
                for (let dx = 0; dx < 6; dx++) {
                    if ((dx + dy + queryLabel.charCodeAt(i)) % 2 === 0) {
                        const px = 50 + i * 8 + dx;
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

        // Save
        const rawPath = join(OUTPUT_DIR, 'decision_map.raw');
        const pngPath = join(OUTPUT_DIR, 'decision_map.png');

        writeFileSync(rawPath, pixels);
        execSync(`convert -size ${width}x${height} -depth 8 rgba:${rawPath} ${pngPath}`);

        console.log(`  ✅ Decision map rendered: ${pngPath}`);
    }
}

// CLI
async function main() {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.log(`
Usage:
  node bin/substrate-executive.js --query "organize the physics documents"
  node bin/substrate-executive.js --query "move biology content north"
  node bin/substrate-executive.js --query "expand the math sector"

Operations:
  MOVE     - Shift content in a direction (north/south/east/west)
  CLUSTER  - Group documents by category
  EXPAND   - Spread content outward
  COMPRESS - Compress content inward
  ALLOCATE - Create new tiles
        `);
        process.exit(0);
    }

    const queryIdx = args.indexOf('--query');
    const query = queryIdx >= 0 ? args.slice(queryIdx + 1).join(' ') : 'analyze the map';

    const executive = new SubstrateExecutive();
    const result = await executive.process(query);

    console.log('\n📊 Executive Summary:');
    console.log(`   Operation: ${result.operation}`);
    console.log(`   Category: ${result.category || 'all'}`);
    console.log(`   Direction: ${result.direction}`);
    console.log(`   Executed: ${result.operationsExecuted} map operations`);
    console.log(`   Attention: ${result.attentionConnections} neural connections`);
}

main().catch(console.error);
