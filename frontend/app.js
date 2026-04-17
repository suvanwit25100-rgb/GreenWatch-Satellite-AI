/* =========================================================================
   GreenWatch.AI — Dashboard Client Logic
   ========================================================================= */

const API = '';  // same-origin; Flask serves frontend

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let selectedFile = null;
let history = [];
let modelInfo = null;
let map = null;
let mapMarker = null;

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const dom = {
    systemStatus:   $('#system-status'),
    statusText:     $('#system-status .status-text'),
    demoBadge:      $('#demo-badge'),
    // Tabs
    tabUpload:      $('#tab-upload'),
    tabRandom:      $('#tab-random'),
    tabMap:         $('#tab-map'),
    contentUpload:  $('#content-upload'),
    contentRandom:  $('#content-random'),
    contentMap:     $('#content-map'),
    // Upload
    uploadZone:     $('#upload-zone'),
    fileInput:      $('#file-input'),
    uploadPreview:  $('#upload-preview'),
    previewImg:     $('#preview-img'),
    clearUpload:    $('#clear-upload'),
    btnPredict:     $('#btn-predict'),
    // Random
    btnRandom:      $('#btn-random'),
    // Results
    resultPanel:     $('#result-panel'),
    resultPlaceholder: $('#result-placeholder'),
    resultContent:   $('#result-content'),
    resultImage:     $('#result-image'),
    resultBadge:     $('#result-label-badge'),
    gaugeCanvas:     $('#confidence-gauge'),
    gaugeLabel:      $('#gauge-label'),
    metaLabel:       $('#meta-label'),
    metaConfidence:  $('#meta-confidence'),
    metaRaw:         $('#meta-raw'),
    metaTime:        $('#meta-time'),
    // Architecture
    archContainer:   $('#arch-container'),
    // History
    historyList:     $('#history-list'),
    historyEmpty:    $('#history-empty'),
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
    setupTabs();
    setupUpload();
    setupButtons();
    createParticles();
    await checkHealth();
    await loadStats();
});

// ---------------------------------------------------------------------------
// Health check
// ---------------------------------------------------------------------------
async function checkHealth() {
    try {
        const res = await fetch(`${API}/api/health`);
        const data = await res.json();

        dom.systemStatus.classList.remove('offline');
        dom.systemStatus.classList.add('online');
        dom.statusText.textContent = 'System Online';

        if (data.demo_mode) {
            dom.demoBadge.classList.remove('hidden');
        }

        if (data.model_info) {
            modelInfo = data.model_info;
            renderArchitecture(modelInfo);
        }
    } catch {
        dom.statusText.textContent = 'Offline';
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------
async function loadStats() {
    try {
        const res = await fetch(`${API}/api/stats`);
        const data = await res.json();

        animateCounter($('#stat-scans .stat-value'), data.total_scans);
        animateCounter($('#stat-forest .stat-value'), data.forest_detected);
        animateCounter($('#stat-alerts .stat-value'), data.deforestation_alerts);
        animateCounter($('#stat-accuracy .stat-value'), data.model_accuracy, '%');
        animateCounter($('#stat-coverage .stat-value'), data.coverage_km2, ' km²');
    } catch {
        // Stats unavailable — leave zeros
    }
}

function animateCounter(el, target, suffix = '') {
    const duration = 1800;
    const start = performance.now();
    const initial = 0;

    function update(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.round(initial + (target - initial) * eased);

        if (suffix === '%') {
            el.innerHTML = `${(initial + (target - initial) * eased).toFixed(1)}<span class="stat-unit">%</span>`;
        } else if (suffix) {
            el.innerHTML = `${current.toLocaleString()}<span class="stat-unit">${suffix}</span>`;
        } else {
            el.textContent = current.toLocaleString();
        }

        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function setupTabs() {
    [dom.tabUpload, dom.tabRandom, dom.tabMap].forEach(tab => {
        if (!tab) return;
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            $$('.tab').forEach(t => t.classList.remove('active'));
            $$('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            $(`#content-${target}`).classList.add('active');
            
            if (target === 'map') {
                initMap();
            }
        });
    });
}

function initMap() {
    if (map) {
        setTimeout(() => map.invalidateSize(), 100);
        return;
    }
    map = L.map('sat-map').setView([20.5937, 78.9629], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);

    map.on('click', async (e) => {
        const {lat, lng} = e.latlng;
        if (mapMarker) map.removeLayer(mapMarker);
        mapMarker = L.marker([lat, lng]).addTo(map);
        await runLocationPrediction(lat, lng);
    });
}

// ---------------------------------------------------------------------------
// Upload
// ---------------------------------------------------------------------------
function setupUpload() {
    const zone = dom.uploadZone;

    zone.addEventListener('click', () => dom.fileInput.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    dom.fileInput.addEventListener('change', () => {
        if (dom.fileInput.files.length) {
            handleFile(dom.fileInput.files[0]);
        }
    });

    dom.clearUpload.addEventListener('click', () => {
        clearFile();
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        dom.previewImg.src = e.target.result;
        dom.uploadPreview.classList.remove('hidden');
        dom.uploadZone.classList.add('hidden');
        dom.btnPredict.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    dom.fileInput.value = '';
    dom.previewImg.src = '';
    dom.uploadPreview.classList.add('hidden');
    dom.uploadZone.classList.remove('hidden');
    dom.btnPredict.disabled = true;
}

// ---------------------------------------------------------------------------
// Predict buttons
// ---------------------------------------------------------------------------
function setupButtons() {
    dom.btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;
        await runPrediction(selectedFile);
    });

    dom.btnRandom.addEventListener('click', async () => {
        await runRandomSample();
    });
}

async function runPrediction(file) {
    dom.btnPredict.disabled = true;
    dom.btnPredict.innerHTML = '<span class="spinner"></span> Analyzing…';

    try {
        const formData = new FormData();
        formData.append('image', file);

        const res = await fetch(`${API.replace('5000', '5001')}/api/predict`, { method: 'POST', body: formData });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        displayResult(data);
        addToHistory(data);
    } catch (err) {
        alert('Failed to connect to the server. Is the backend running?');
        console.error(err);
    } finally {
        dom.btnPredict.disabled = false;
        dom.btnPredict.innerHTML = '<span class="btn-icon">🛰️</span> Analyze Image';
    }
}

async function runRandomSample() {
    dom.btnRandom.disabled = true;
    dom.btnRandom.innerHTML = '<span class="spinner"></span> Fetching…';

    try {
        const res = await fetch(`${API}/api/random-sample`);
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        displayResult(data);
        addToHistory(data);
    } catch (err) {
        alert('Failed to connect to the server.');
        console.error(err);
    } finally {
        dom.btnRandom.disabled = false;
        dom.btnRandom.innerHTML = '<span class="btn-icon">🎲</span> Analyze Random Sample';
    }
}

async function runLocationPrediction(lat, lng) {
    dom.resultPlaceholder.classList.remove('hidden');
    dom.resultContent.classList.add('hidden');
    dom.resultPlaceholder.innerHTML = '<div class="placeholder-icon">🛰️</div><p>Scanning coordinates...</p><p class="placeholder-hint">Downloading raw tile imagery from external satellite API...</p>';

    try {
        const res = await fetch(`${API.replace('5000', '5001')}/api/predict-location?lat=${lat}&lng=${lng}`);
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        displayResult(data);
        addToHistory(data);
    } catch (err) {
        alert('Failed to connect to the server.');
        console.error(err);
    } finally {
        setTimeout(() => {
            dom.resultPlaceholder.innerHTML = '<div class="placeholder-icon">🛰️</div><p>Waiting for satellite imagery…</p><p class="placeholder-hint">Upload an image or test a random sample to begin analysis</p>';
        }, 500);
    }
}

// ---------------------------------------------------------------------------
// Display result
// ---------------------------------------------------------------------------
function displayResult(data) {
    dom.resultPlaceholder.classList.add('hidden');
    dom.resultContent.classList.remove('hidden');

    // Image
    if (data.image_b64) {
        dom.resultImage.src = `data:image/jpeg;base64,${data.image_b64}`;
    } else {
        // No image available (pure demo)
        dom.resultImage.src = '';
        dom.resultImage.alt = 'Demo mode — no image available';
    }

    // Label badge
    const isForest = data.label === 'Forest';
    dom.resultBadge.textContent = isForest ? '🌲 Forest' : '🔥 Deforested';
    dom.resultBadge.className = 'result-label-badge ' + (isForest ? 'forest' : 'deforested');

    // Meta
    dom.metaLabel.textContent = data.label;
    dom.metaLabel.style.color = isForest ? 'var(--green)' : 'var(--red)';
    dom.metaConfidence.textContent = `${data.confidence}%`;
    dom.metaRaw.textContent = data.raw_score;
    dom.metaTime.textContent = `${data.inference_ms} ms`;

    // Gauge
    drawGauge(data.confidence, isForest);
    dom.gaugeLabel.textContent = 'Confidence Score';

    // Scroll into view
    dom.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ---------------------------------------------------------------------------
// Confidence Gauge (Canvas)
// ---------------------------------------------------------------------------
function drawGauge(percent, isForest) {
    const canvas = dom.gaugeCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // HiDPI
    canvas.width = 220 * dpr;
    canvas.height = 130 * dpr;
    canvas.style.width = '220px';
    canvas.style.height = '130px';
    ctx.scale(dpr, dpr);

    const cx = 110;
    const cy = 110;
    const radius = 85;
    const lineWidth = 14;
    const startAngle = Math.PI;
    const endAngle = 2 * Math.PI;
    const targetAngle = startAngle + (percent / 100) * Math.PI;

    ctx.clearRect(0, 0, 220, 130);

    // Track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Animate fill
    let currentAngle = startAngle;
    const color = isForest ? '#10b981' : '#ef4444';
    const glowColor = isForest ? 'rgba(16,185,129,0.4)' : 'rgba(239,68,68,0.4)';

    function animateGauge() {
        currentAngle += (targetAngle - currentAngle) * 0.08;

        ctx.clearRect(0, 0, 220, 130);

        // Track
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, endAngle);
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Glow
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, currentAngle);
        ctx.strokeStyle = glowColor;
        ctx.lineWidth = lineWidth + 8;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Fill
        ctx.beginPath();
        ctx.arc(cx, cy, radius, startAngle, currentAngle);
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.stroke();

        // Center text
        ctx.fillStyle = color;
        ctx.font = `bold 32px 'Inter', sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const displayPercent = Math.round(((currentAngle - startAngle) / Math.PI) * 100);
        ctx.fillText(`${displayPercent}%`, cx, cy - 16);

        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        ctx.font = `500 11px 'Inter', sans-serif`;
        ctx.fillText(isForest ? 'FOREST' : 'DEFORESTED', cx, cy + 10);

        if (Math.abs(currentAngle - targetAngle) > 0.005) {
            requestAnimationFrame(animateGauge);
        }
    }
    animateGauge();
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------
function addToHistory(data) {
    dom.historyEmpty.classList.add('hidden');

    const item = document.createElement('div');
    item.className = 'history-item';
    const isForest = data.label === 'Forest';
    const timeStr = new Date().toLocaleTimeString();

    item.innerHTML = `
        ${data.image_b64 ? `<img class="history-thumb" src="data:image/jpeg;base64,${data.image_b64}" alt="Thumbnail" />` : ''}
        <div class="history-info">
            <div class="history-label ${isForest ? 'forest' : 'deforested'}">
                ${isForest ? '🌲' : '🔥'} ${data.label}
            </div>
            <div class="history-conf">${data.confidence}% · ${data.inference_ms} ms${data.filename ? ` · ${data.filename}` : ''}</div>
        </div>
        <div class="history-time">${timeStr}</div>
    `;

    // Insert at top
    if (dom.historyList.firstChild && dom.historyList.firstChild !== dom.historyEmpty) {
        dom.historyList.insertBefore(item, dom.historyList.firstChild);
    } else {
        dom.historyList.appendChild(item);
    }

    // Keep max 10
    const items = dom.historyList.querySelectorAll('.history-item');
    if (items.length > 10) {
        items[items.length - 1].remove();
    }

    history.push(data);
}

// ---------------------------------------------------------------------------
// Architecture Visualizer
// ---------------------------------------------------------------------------
function renderArchitecture(info) {
    if (!info || !info.layers) return;

    const pipeline = document.createElement('div');
    pipeline.className = 'arch-pipeline';

    info.layers.forEach((layer, i) => {
        const node = document.createElement('div');
        node.className = 'arch-layer';
        node.setAttribute('data-type', layer.type);
        node.innerHTML = `
            <div class="arch-layer-name">${layer.name}</div>
            <div class="arch-layer-detail">${layer.detail}</div>
        `;
        pipeline.appendChild(node);

        // Arrow between layers
        if (i < info.layers.length - 1) {
            const arrow = document.createElement('div');
            arrow.className = 'arch-arrow';
            arrow.innerHTML = '→';
            pipeline.appendChild(arrow);
        }
    });

    dom.archContainer.innerHTML = '';
    dom.archContainer.appendChild(pipeline);

    // Meta info below
    const meta = document.createElement('div');
    meta.style.cssText = 'display:flex; gap:24px; justify-content:center; margin-top:24px; flex-wrap:wrap;';
    meta.innerHTML = `
        <div class="meta-item" style="min-width:auto; flex:0;">
            <span class="meta-key">Optimizer</span>
            <span class="meta-value mono">${info.optimizer}</span>
        </div>
        <div class="meta-item" style="min-width:auto; flex:0;">
            <span class="meta-key">Loss Function</span>
            <span class="meta-value mono">${info.loss}</span>
        </div>
        <div class="meta-item" style="min-width:auto; flex:0;">
            <span class="meta-key">Input Shape</span>
            <span class="meta-value mono">${info.input_shape}</span>
        </div>
        <div class="meta-item" style="min-width:auto; flex:0;">
            <span class="meta-key">Epochs</span>
            <span class="meta-value mono">${info.training_epochs}</span>
        </div>
    `;
    dom.archContainer.appendChild(meta);
}

// ---------------------------------------------------------------------------
// Hero Particles
// ---------------------------------------------------------------------------
function createParticles() {
    const container = $('#hero-particles');
    if (!container) return;

    for (let i = 0; i < 30; i++) {
        const dot = document.createElement('div');
        dot.style.cssText = `
            position: absolute;
            width: ${2 + Math.random() * 3}px;
            height: ${2 + Math.random() * 3}px;
            background: rgba(16, 185, 129, ${0.15 + Math.random() * 0.25});
            border-radius: 50%;
            top: ${Math.random() * 100}%;
            left: ${Math.random() * 100}%;
            animation: float ${3 + Math.random() * 4}s ease-in-out infinite;
            animation-delay: ${Math.random() * 3}s;
        `;
        container.appendChild(dot);
    }
}
