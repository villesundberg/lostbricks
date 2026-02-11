// LostBricks — scan.js
// Camera capture, segmentation on annotated photo, color-sorted part picker

// ── IndexedDB for crop storage ──

const DB_NAME = 'lostbricks-crops';
const DB_VERSION = 1;

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('crops')) {
        db.createObjectStore('crops', { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function storeCrop(setNum, partNum, colorId, imageBlob) {
  const db = await openDB();
  const tx = db.transaction('crops', 'readwrite');
  tx.objectStore('crops').put({ setNum, partNum, colorId, imageBlob, timestamp: Date.now() });
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

// ── Segmentation ──

function segmentImage(canvas) {
  const ctx = canvas.getContext('2d');
  const { width, height } = canvas;
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const totalPixels = width * height;

  const threshold = learnedThreshold || 200;
  console.log(`segmentation threshold: ${threshold}${learnedThreshold ? ' (learned)' : ' (default)'}`);

  const SAT_THRESHOLD = 0.15; // recover bright-but-colorful pixels (yellow, tan)
  const mask = new Uint8Array(totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
    if (brightness < threshold) {
      mask[i] = 1;
    } else {
      // Bright pixel — check if it's colorful (saturated) rather than white background
      const mx = Math.max(r, g, b);
      const sat = mx > 0 ? (mx - Math.min(r, g, b)) / mx : 0;
      mask[i] = sat > SAT_THRESHOLD ? 1 : 0;
    }
  }

  // Connected components via BFS
  const labels = new Int32Array(totalPixels);
  let nextLabel = 1;
  const components = [];

  for (let i = 0; i < totalPixels; i++) {
    if (mask[i] === 0 || labels[i] !== 0) continue;

    const label = nextLabel++;
    let minX = width, minY = height, maxX = 0, maxY = 0;
    let pixelCount = 0;
    let totalR = 0, totalG = 0, totalB = 0;
    const histR = new Float64Array(16);
    const histG = new Float64Array(16);
    const histB = new Float64Array(16);
    const queue = [i];
    labels[i] = label;

    while (queue.length > 0) {
      const idx = queue.pop();
      const x = idx % width;
      const y = (idx - x) / width;

      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      pixelCount++;
      const pr = data[idx * 4], pg = data[idx * 4 + 1], pb = data[idx * 4 + 2];
      totalR += pr;
      totalG += pg;
      totalB += pb;
      histR[Math.min(15, pr >> 4)]++;
      histG[Math.min(15, pg >> 4)]++;
      histB[Math.min(15, pb >> 4)]++;

      const neighbors = [];
      if (x > 0) neighbors.push(idx - 1);
      if (x < width - 1) neighbors.push(idx + 1);
      if (y > 0) neighbors.push(idx - width);
      if (y < height - 1) neighbors.push(idx + width);

      for (const n of neighbors) {
        if (mask[n] === 1 && labels[n] === 0) {
          labels[n] = label;
          queue.push(n);
        }
      }
    }

    components.push({
      label, minX, minY, maxX, maxY, pixelCount,
      avgR: Math.round(totalR / pixelCount),
      avgG: Math.round(totalG / pixelCount),
      avgB: Math.round(totalB / pixelCount),
      histR: Array.from(histR),
      histG: Array.from(histG),
      histB: Array.from(histB),
    });
  }

  // Filter: min 0.2% of image, max 25% (reject background)
  const minArea = totalPixels * 0.002;
  const maxArea = totalPixels * 0.25;
  return components
    .filter((c) => c.pixelCount >= minArea && c.pixelCount <= maxArea)
    .sort((a, b) => b.pixelCount - a.pixelCount);
}

// Perceptually weighted color distance (redmean approximation)
function colorDist(r1, g1, b1, r2, g2, b2) {
  const rmean = (r1 + r2) / 2;
  const dr = r1 - r2, dg = g1 - g2, db = b1 - b2;
  return Math.sqrt((2 + rmean / 256) * dr * dr + 4 * dg * dg + (2 + (255 - rmean) / 256) * db * db);
}

// Histogram distance: L2 between normalized distributions
function histDist(h1, h2) {
  const s1 = h1.reduce((a, b) => a + b, 0) || 1;
  const s2 = h2.reduce((a, b) => a + b, 0) || 1;
  let d = 0;
  for (let i = 0; i < h1.length; i++) {
    const diff = h1[i] / s1 - h2[i] / s2;
    d += diff * diff;
  }
  return Math.sqrt(d);
}

// Combined RGB histogram distance
function colorHistDist(blob, learnedHist) {
  if (!blob.histR || !learnedHist.histR) return Infinity;
  return histDist(blob.histR, learnedHist.histR) +
         histDist(blob.histG, learnedHist.histG) +
         histDist(blob.histB, learnedHist.histB);
}

function hexToRGB(hex) {
  const n = parseInt(hex, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ── Learning API ──

function blobToBase64(blob) {
  const pad = 8;
  const x = Math.max(0, blob.minX - pad);
  const y = Math.max(0, blob.minY - pad);
  const w = Math.min(capturedCanvas.width, blob.maxX + pad) - x;
  const h = Math.min(capturedCanvas.height, blob.maxY + pad) - y;
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  tmp.getContext('2d').drawImage(capturedCanvas, x, y, w, h, 0, 0, w, h);
  // Strip "data:image/jpeg;base64," prefix
  return tmp.toDataURL('image/jpeg', 0.8).split(',')[1];
}

function canvasToBase64(canvas) {
  return canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
}

async function apiSaveSession() {
  if (!capturedCanvas || !rawCanvas) return;
  const setNum = window.lostbricks.getCurrentSetNum();
  const stripBlob = (b) => ({
    minX: b.minX, minY: b.minY, maxX: b.maxX, maxY: b.maxY,
    avgR: b.avgR, avgG: b.avgG, avgB: b.avgB,
    histR: b.histR, histG: b.histG, histB: b.histB,
    source: b.source || 'auto',
    partKey: b.partKey, partName: b.partName,
  });
  const payload = {
    setNum,
    timestamp: Date.now(),
    rawPhoto: canvasToBase64(rawCanvas),
    cropRect: { ...cropRect },
    croppedPhoto: canvasToBase64(capturedCanvas),
    blobs: blobs.map(stripBlob),
    deletedBlobs: deletedBlobs.map(stripBlob),
  };
  try {
    const res = await fetch('api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.ok) console.log(`session saved: ${data.path}`);
    else console.warn('session save error:', data.error);
  } catch (e) {
    console.warn('session save failed:', e);
  }
}

async function apiLabel(setNum, partNum, colorId, jpegB64, avgR, avgG, avgB, histR, histG, histB) {
  try {
    const res = await fetch('api/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setNum, partNum, colorId: String(colorId), jpeg: jpegB64, avgR, avgG, avgB, histR, histG, histB }),
    });
    const data = await res.json();
    if (data.ok) {
      // Update local learned colors cache immediately
      if (data.learnedColors) learnedColors = data.learnedColors;
      if (data.cropCount) learnedCropCount = data.cropCount;
      console.log(`learn: ${partNum}_${colorId} (${data.cropCount} crops)`);
    }
    else console.warn('learn error:', data.error);
  } catch (e) {
    console.warn('learn send failed:', e);
  }
}

async function apiReject(setNum, wrongKey, jpegB64) {
  try {
    await fetch('api/reject', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setNum, wrongKey, jpeg: jpegB64 }),
    });
  } catch (e) {
    console.warn('reject send failed:', e);
  }
}

async function apiSuggest(setNum, jpegB64) {
  try {
    const res = await fetch('api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setNum, jpeg: jpegB64 }),
    });
    const data = await res.json();
    return {
      suggestions: data.suggestions || [],
      learnedColors: data.learnedColors || {},
      cropCount: data.cropCount || 0,
      threshold: data.threshold || null,
      colorProbs: data.colorProbs || {},
    };
  } catch (e) {
    console.warn('suggest failed:', e);
    return { suggestions: [], learnedColors: {}, cropCount: 0, colorProbs: {} };
  }
}

async function bootstrapCatalog() {
  const setNum = window.lostbricks.getCurrentSetNum();
  if (!setNum) return;
  const parts = window.lostbricks.getCurrentSetParts();
  if (!parts.length) return;
  const payload = parts.map(p => ({
    key: `${p.partNum}_${p.color.id}`,
    imgUrl: p.imgUrl,
  })).filter(p => p.imgUrl);
  try {
    const res = await fetch('api/bootstrap', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setNum, parts: payload }),
    });
    const data = await res.json();
    if (data.downloading) {
      console.log(`bootstrap: downloading ${data.downloading} catalog images`);
    }
  } catch (e) {
    console.warn('bootstrap failed:', e);
  }
}

// Cached learned data (updated with each suggest call)
let learnedColors = {}; // { colorId: { r, g, b, count } }
let learnedCropCount = 0;
let learnedThreshold = null; // learned brightness threshold for segmentation

async function fetchLearnedParams() {
  const setNum = window.lostbricks.getCurrentSetNum();
  if (!setNum) return;
  try {
    const res = await fetch(`api/learned/${setNum}`);
    const data = await res.json();
    if (data.colors) learnedColors = data.colors;
    learnedCropCount = data.cropCount || 0;
    if (data.threshold) {
      learnedThreshold = data.threshold;
      console.log(`learned threshold: ${learnedThreshold} (from server)`);
    }
  } catch (e) {
    console.warn('fetch learned failed:', e);
  }
}

// ── Camera ──

let stream = null;

async function startCamera() {
  const video = document.getElementById('scan-video');
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: false,
    });
    video.srcObject = stream;
  } catch (err) {
    alert('Camera access failed: ' + err.message);
    closeScan();
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  document.getElementById('scan-video').srcObject = null;
}

function captureFrame() {
  const video = document.getElementById('scan-video');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  return canvas;
}

// ── UI State ──

let blobs = []; // { minX, minY, maxX, maxY, avgR, avgG, avgB, partKey, partName, source }
let deletedBlobs = []; // blobs the user explicitly removed
let capturedCanvas = null; // cropped frame
let selectedBlobIdx = null;
let pickerApplyAll = false; // when true, assignPart labels ALL blobs

function openScan() {
  document.getElementById('scan-modal').classList.remove('hidden');
  document.getElementById('scan-viewfinder').classList.remove('hidden');
  document.getElementById('scan-crop').classList.add('hidden');
  document.getElementById('scan-annotate').classList.add('hidden');
  document.getElementById('part-picker').classList.add('hidden');
  blobs = [];
  deletedBlobs = [];
  selectedBlobIdx = null;
  capturedCanvas = null;
  clearFloatingLabels();
  fetchLearnedParams(); // get latest threshold + colors from server
  bootstrapCatalog(); // ensure catalog images are in the index
  startCamera();
}

function closeScan() {
  stopCamera();
  document.getElementById('scan-modal').classList.add('hidden');
}

function showAnnotated(detected) {
  blobs = detected.map((c) => ({
    ...c, partKey: null, partName: null, source: 'auto',
  }));
  stopCamera();
  document.getElementById('scan-viewfinder').classList.add('hidden');
  document.getElementById('scan-annotate').classList.remove('hidden');
  updateStatus();
  drawAnnotatedPhoto();
  updateFloatingLabels();
}

function updateStatus() {
  const labeled = blobs.filter((b) => b.partKey).length;
  document.getElementById('scan-status').textContent = `${labeled} / ${blobs.length} labeled`;
}

// ── Annotated Photo Drawing ──

function drawAnnotatedPhoto() {
  const canvas = document.getElementById('scan-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = capturedCanvas.width;
  canvas.height = capturedCanvas.height;

  // Draw original photo
  ctx.drawImage(capturedCanvas, 0, 0);

  const pad = 8;
  blobs.forEach((blob, i) => {
    const x = Math.max(0, blob.minX - pad);
    const y = Math.max(0, blob.minY - pad);
    const w = Math.min(canvas.width, blob.maxX + pad) - x;
    const h = Math.min(canvas.height, blob.maxY + pad) - y;

    if (blob.partKey) {
      // Labeled — green
      ctx.strokeStyle = '#2ecc71';
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, w, h);
      // Leader line to floating label
      if (blob.labelOffX !== undefined) {
        const bcx = (blob.minX + blob.maxX) / 2;
        const bcy = (blob.minY + blob.maxY) / 2;
        const lx = bcx + blob.labelOffX;
        const ly = bcy + blob.labelOffY;
        ctx.save();
        ctx.strokeStyle = 'rgba(46, 204, 113, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath();
        ctx.moveTo(bcx, bcy);
        ctx.lineTo(lx, ly);
        ctx.stroke();
        ctx.restore();
      }
    } else if (i === selectedBlobIdx) {
      // Selected — accent
      ctx.strokeStyle = '#e94560';
      ctx.lineWidth = 5;
      ctx.strokeRect(x, y, w, h);
    } else {
      // Unlabeled — yellow
      ctx.strokeStyle = '#f1c40f';
      ctx.lineWidth = 3;
      ctx.setLineDash([8, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
      // Number label
      ctx.font = 'bold 28px sans-serif';
      ctx.fillStyle = '#f1c40f';
      ctx.fillText(String(i + 1), x + 4, y + 30);
    }
  });
}

// ── Canvas Tap Handling ──

function getCanvasTapPosition(e) {
  const canvas = document.getElementById('scan-canvas');
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  let clientX, clientY;
  if (e.touches) {
    clientX = e.touches[0].clientX;
    clientY = e.touches[0].clientY;
  } else {
    clientX = e.clientX;
    clientY = e.clientY;
  }

  return {
    x: (clientX - rect.left) * scaleX,
    y: (clientY - rect.top) * scaleY,
  };
}

function findBlobAt(px, py) {
  const pad = 8;
  // Check smallest blobs first (more specific)
  const sorted = blobs.map((b, i) => ({ b, i })).sort((a, b) => a.b.pixelCount - b.b.pixelCount);
  for (const { b, i } of sorted) {
    if (px >= b.minX - pad && px <= b.maxX + pad && py >= b.minY - pad && py <= b.maxY + pad) {
      return i;
    }
  }
  return -1;
}

// ── Manual bounding box drawing ──

let drawState = null; // { startX, startY } while dragging

function canvasPointerDown(e) {
  e.preventDefault();
  const pos = getCanvasTapPosition(e);
  const idx = findBlobAt(pos.x, pos.y);

  if (idx >= 0) {
    // Tapped an existing blob
    selectedBlobIdx = idx;
    drawAnnotatedPhoto();
    showPartPicker(blobs[idx]);
    drawState = null;
    return;
  }

  // Start drawing a manual bounding box
  drawState = { startX: pos.x, startY: pos.y };
}

function canvasPointerMove(e) {
  if (!drawState) return;
  e.preventDefault();
  const pos = getCanvasTapPosition(e);

  // Redraw photo + existing annotations + live rectangle
  drawAnnotatedPhoto();
  const canvas = document.getElementById('scan-canvas');
  const ctx = canvas.getContext('2d');
  const x = Math.min(drawState.startX, pos.x);
  const y = Math.min(drawState.startY, pos.y);
  const w = Math.abs(pos.x - drawState.startX);
  const h = Math.abs(pos.y - drawState.startY);
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth = 3;
  ctx.setLineDash([6, 3]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
}

function canvasPointerUp(e) {
  if (!drawState) return;
  const pos = getCanvasTapPosition(e);
  const x1 = Math.min(drawState.startX, pos.x);
  const y1 = Math.min(drawState.startY, pos.y);
  const x2 = Math.max(drawState.startX, pos.x);
  const y2 = Math.max(drawState.startY, pos.y);
  drawState = null;

  const w = x2 - x1;
  const h = y2 - y1;

  // Ignore tiny drags (accidental taps)
  if (w < 20 && h < 20) {
    // Just a tap on empty space — deselect
    hidePartPicker();
    selectedBlobIdx = null;
    drawAnnotatedPhoto();
    return;
  }

  // Sample average color + histograms from foreground pixels only
  const ctx = capturedCanvas.getContext('2d');
  const imageData = ctx.getImageData(Math.round(x1), Math.round(y1), Math.round(w), Math.round(h));
  const data = imageData.data;
  const totalPx = data.length / 4;
  let totalR = 0, totalG = 0, totalB = 0, fgCount = 0;
  const mHistR = new Float64Array(16);
  const mHistG = new Float64Array(16);
  const mHistB = new Float64Array(16);
  const manualThresh = learnedThreshold || 200;
  for (let i = 0; i < totalPx; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
    const mx = Math.max(r, g, b);
    const sat = mx > 0 ? (mx - Math.min(r, g, b)) / mx : 0;
    if (brightness < manualThresh || sat > 0.15) {
      totalR += r;
      totalG += g;
      totalB += b;
      mHistR[Math.min(15, r >> 4)]++;
      mHistG[Math.min(15, g >> 4)]++;
      mHistB[Math.min(15, b >> 4)]++;
      fgCount++;
    }
  }
  if (fgCount === 0) fgCount = 1; // fallback

  // Add as a new blob
  const newBlob = {
    minX: Math.round(x1), minY: Math.round(y1),
    maxX: Math.round(x2), maxY: Math.round(y2),
    pixelCount: Math.round(w * h),
    avgR: Math.round(totalR / fgCount),
    avgG: Math.round(totalG / fgCount),
    avgB: Math.round(totalB / fgCount),
    histR: Array.from(mHistR),
    histG: Array.from(mHistG),
    histB: Array.from(mHistB),
    partKey: null, partName: null, source: 'manual',
  };
  blobs.push(newBlob);
  selectedBlobIdx = null; // don't auto-select; user taps when ready
  updateStatus();
  drawAnnotatedPhoto();
}

const scanCanvas = document.getElementById('scan-canvas');
scanCanvas.addEventListener('pointerdown', canvasPointerDown);
scanCanvas.addEventListener('pointermove', canvasPointerMove);
scanCanvas.addEventListener('pointerup', canvasPointerUp);

// ── Part Picker ──

let pickerBlob = null;
let pickerParts = []; // deduplicated, scored
let pickerColorFilter = null; // null = all, or color id

function buildColorChips() {
  const colorMap = new Map();
  for (const p of pickerParts) {
    if (!colorMap.has(p.color.id)) {
      colorMap.set(p.color.id, { name: p.color.name, rgb: p.color.rgb, dist: p.dist || 0 });
    }
  }
  const sortedColors = [...colorMap.entries()].sort((a, b) => a[1].dist - b[1].dist);

  // Pre-select best matching color if not already set
  if (pickerColorFilter === null && sortedColors.length > 0) {
    pickerColorFilter = sortedColors[0][0];
  }

  const colorsEl = document.getElementById('picker-colors');
  colorsEl.innerHTML = `<div class="picker-color-chip ${pickerColorFilter === null ? 'active' : ''}" data-color="all">All</div>` +
    sortedColors.map(([id, c]) =>
      `<div class="picker-color-chip ${id === pickerColorFilter ? 'active' : ''}" data-color="${id}">
        <span class="color-swatch" style="background:#${c.rgb}"></span>${c.name}
      </div>`
    ).join('');

  colorsEl.querySelectorAll('.picker-color-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const val = chip.dataset.color;
      pickerColorFilter = val === 'all' ? null : parseInt(val);
      colorsEl.querySelectorAll('.picker-color-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      renderPickerGrid();
    });
  });
}

function showPartPicker(blob, allowAutoAssign = false) {
  pickerColorFilter = null; // reset for each new blob
  pickerBlob = blob;
  const parts = window.lostbricks.getCurrentSetParts();

  // Deduplicate
  const seen = new Set();
  pickerParts = parts.filter((p) => {
    const key = `${p.partNum}_${p.color.id}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });

  // Show blob crop in header
  const blobImg = document.getElementById('picker-blob-img');
  let jpegB64 = null;
  if (blob && capturedCanvas) {
    jpegB64 = blobToBase64(blob);
    blobImg.src = `data:image/jpeg;base64,${jpegB64}`;
  } else {
    blobImg.src = '';
  }

  // Compute color priors from remaining parts in this set
  // P(color) = fraction of remaining pieces that are this color
  const colorRemaining = {};
  let totalRemaining = 0;
  for (const p of pickerParts) {
    const cid = String(p.color.id);
    const have = window.lostbricks.getHave(`${p.partNum}_${p.color.id}`);
    const rem = Math.max(0, p.qty - have);
    colorRemaining[cid] = (colorRemaining[cid] || 0) + rem;
    totalRemaining += rem;
  }
  const colorPrior = {};
  for (const [cid, count] of Object.entries(colorRemaining)) {
    colorPrior[cid] = totalRemaining > 0 ? count / totalRemaining : 0;
  }

  // Score by color distance + remaining need
  // Use histogram distance when available, fall back to avg RGB distance
  if (blob) {
    const { avgR, avgG, avgB } = blob;
    const hasHist = blob.histR && blob.histR.length === 16;
    pickerParts = pickerParts.map((p) => {
      const key = `${p.partNum}_${p.color.id}`;
      const lc = learnedColors[String(p.color.id)];
      let colorScore, dist;

      if (hasHist && lc && lc.histR) {
        // Histogram distance (captures distribution shape, resistant to outliers)
        dist = colorHistDist(blob, lc);
        colorScore = dist * 200; // scale to comparable range with old colorDist
      } else if (lc && lc.count >= 1) {
        // Fall back to avg RGB distance with learned colors
        dist = colorDist(avgR, avgG, avgB, lc.r, lc.g, lc.b);
        colorScore = dist;
        if (dist < 60) colorScore = dist * 0.3; // close colors: let ML decide
      } else {
        // Fall back to catalog colors (unreliable)
        const [pr, pg, pb] = hexToRGB(p.color.rgb);
        dist = colorDist(avgR, avgG, avgB, pr, pg, pb);
        colorScore = dist * 0.3;
      }

      const have = window.lostbricks.getHave(key);
      const remaining = Math.max(0, p.qty - have);
      const score = colorScore - remaining * 10;
      return { ...p, dist, remaining, score, mlBoost: 0 };
    }).sort((a, b) => a.score - b.score);
  }

  // Build color chips + render immediately (before ML results)
  buildColorChips();
  renderPickerGrid();
  document.getElementById('picker-colors').scrollLeft = 0;

  // Ask server for ML suggestions (async — updates grid when ready)
  if (jpegB64) {
    const setNum = window.lostbricks.getCurrentSetNum();
    apiSuggest(setNum, jpegB64).then((result) => {
      const { suggestions, learnedColors: lc, cropCount } = result;

      // Update cached learned data
      if (lc) learnedColors = lc;
      learnedCropCount = cropCount;
      if (result.threshold) learnedThreshold = result.threshold;

      if (suggestions.length === 0) {
        updatePickerTitle();
        return;
      }

      // Apply ML boost to matching parts (decays with rank, always positive)
      const boostMap = new Map();
      suggestions.forEach((s, i) => {
        boostMap.set(s.key, 100 / (1 + i * 0.5));
      });
      for (const p of pickerParts) {
        const key = `${p.partNum}_${p.color.id}`;
        const boost = boostMap.get(key);
        if (boost) {
          p.mlBoost = boost;
          p.score -= boost;
        }
      }

      // Derive color probabilities from k-NN results (CNN already encodes color)
      const knnColorVotes = {};
      for (const s of suggestions) {
        const cid = s.key.split('_')[1];
        const weight = 1 / (1 + s.dist);
        knnColorVotes[cid] = (knnColorVotes[cid] || 0) + weight;
      }
      // Apply Bayesian prior from set composition
      let posteriorSum = 0;
      const posteriors = {};
      for (const cid of Object.keys(knnColorVotes)) {
        const likelihood = knnColorVotes[cid];
        const prior = colorPrior[cid] || 0.001;
        posteriors[cid] = likelihood * prior;
        posteriorSum += posteriors[cid];
      }
      if (posteriorSum > 0) {
        for (const cid of Object.keys(posteriors)) {
          posteriors[cid] /= posteriorSum;
        }
      }
      for (const p of pickerParts) {
        const cid = String(p.color.id);
        const posterior = posteriors[cid] || 0;
        p.colorProb = posterior;
        p.score -= posterior * 150;
      }
      pickerParts.sort((a, b) => a.score - b.score);

      // If top suggestion is very confident, auto-assign (only when not user-initiated)
      const topDist = suggestions[0].dist;
      const secondDist = suggestions.length > 1 ? suggestions[1].dist : 999;
      const gap = secondDist - topDist;
      const topKey = suggestions[0].key;
      const topPart = pickerParts.find((p) => `${p.partNum}_${p.color.id}` === topKey);

      if (allowAutoAssign && topDist < 0.3 && gap > 0.15 && topPart && selectedBlobIdx !== null && !blob.partKey) {
        console.log(`AUTO: ${topKey} (d=${topDist}, gap=${gap.toFixed(3)})`);
        blob.autoAssignedKey = topKey; // track for rejection learning
        assignPart(topKey, topPart.name);
        return;
      }

      // Otherwise just auto-select the color tab for the best match
      if (topDist < 0.5 && topPart) {
        pickerColorFilter = topPart.color.id;
      }

      buildColorChips();
      renderPickerGrid();
      updatePickerTitle();
      const topColorId = topKey.split('_')[1];
      const postP = topPart ? (topPart.colorProb || 0).toFixed(2) : '?';
      console.log(`ML: ${cropCount} crops, top=${topKey} (d=${topDist}, gap=${gap.toFixed(3)}, colorPost=${postP})`);
    });
  }

  updatePickerTitle();
  document.getElementById('part-picker').classList.remove('hidden');
}

function updatePickerTitle() {
  const title = document.getElementById('picker-title');
  if (learnedCropCount > 0) {
    const colorCount = Object.keys(learnedColors).length;
    title.textContent = `Pick a part (${learnedCropCount} learned, ${colorCount} colors)`;
  } else {
    title.textContent = 'Pick a part';
  }
}

function renderPickerGrid() {
  const grid = document.getElementById('picker-grid');
  let filtered = pickerParts;
  if (pickerColorFilter !== null) {
    filtered = pickerParts.filter((p) => p.color.id === pickerColorFilter);
  }

  grid.innerHTML = filtered.map((p, i) => {
    const key = `${p.partNum}_${p.color.id}`;
    const remaining = p.remaining !== undefined ? p.remaining : '';
    const isTopML = p.mlBoost > 50;
    const cls = `picker-item${isTopML ? ' ml-top' : ''}`;
    return `
      <div class="${cls}" data-key="${key}" data-name="${p.name}">
        <img src="${p.imgUrl || ''}" alt="" loading="lazy">
        <div class="picker-label">${p.name}${remaining ? ` (${remaining})` : ''}${isTopML ? ' *' : ''}</div>
      </div>
    `;
  }).join('');

  grid.querySelectorAll('.picker-item').forEach((item) => {
    item.addEventListener('click', () => assignPart(item.dataset.key, item.dataset.name));
  });

  grid.scrollTop = 0;
}

function hidePartPicker() {
  document.getElementById('part-picker').classList.add('hidden');
}

async function assignPart(key, name) {
  if (selectedBlobIdx === null && !pickerApplyAll) return;

  const targetBlobs = pickerApplyAll ? blobs : [blobs[selectedBlobIdx]];
  const setNum = window.lostbricks.getCurrentSetNum();
  const [partNum, colorId] = key.split('_');

  for (const blob of targetBlobs) {
    // Learn from rejected auto-assign
    if (blob.autoAssignedKey && blob.autoAssignedKey !== key) {
      console.log(`REJECT auto: ${blob.autoAssignedKey} → ${key}`);
      apiReject(setNum, blob.autoAssignedKey, blobToBase64(blob));
    }
    blob.autoAssignedKey = null;

    // Just update the blob label — counts are applied in batch on "Done"
    blob.partKey = key;
    blob.partName = name;

    // Send crop to learning server (fire-and-forget)
    const jpegB64 = blobToBase64(blob);
    apiLabel(setNum, partNum, colorId, jpegB64, blob.avgR, blob.avgG, blob.avgB, blob.histR, blob.histG, blob.histB);

    // Also store in IndexedDB
    try {
      const jpeg = await fetch(`data:image/jpeg;base64,${jpegB64}`).then((r) => r.blob());
      await storeCrop(setNum, partNum, parseInt(colorId), jpeg);
    } catch (e) {
      console.warn('Failed to store crop:', e);
    }
  }

  pickerApplyAll = false;
  hidePartPicker();
  selectedBlobIdx = null;
  updateStatus();
  initLabelPositions();
  drawAnnotatedPhoto();
  updateFloatingLabels();
}

// ── Floating Labels ──

function getPartImgUrl(partKey) {
  const parts = window.lostbricks.getCurrentSetParts();
  const [partNum, colorId] = partKey.split('_');
  const part = parts.find(p => p.partNum === partNum && String(p.color.id) === colorId);
  return part ? part.imgUrl || '' : '';
}

function canvasToWrapCSS(cx, cy) {
  const canvas = document.getElementById('scan-canvas');
  const wrap = document.getElementById('scan-photo-wrap');
  const cRect = canvas.getBoundingClientRect();
  const wRect = wrap.getBoundingClientRect();
  return {
    x: (cRect.left - wRect.left) + cx * (cRect.width / canvas.width),
    y: (cRect.top - wRect.top) + cy * (cRect.height / canvas.height),
  };
}

const LABEL_EST_W = 140; // CSS px estimates for overlap avoidance
const LABEL_EST_H = 46;

function initLabelPositions() {
  const canvas = document.getElementById('scan-canvas');
  const cRect = canvas.getBoundingClientRect();
  if (!cRect.width) return;
  const sx = cRect.width / canvas.width;
  const sy = cRect.height / canvas.height;

  // Set initial offset for newly labeled blobs
  for (const blob of blobs) {
    if (blob.partKey && blob.labelOffX === undefined) {
      blob.labelOffX = 0;
      blob.labelOffY = -((blob.maxY - blob.minY) / 2 + LABEL_EST_H / sy + 10);
    }
  }

  // Overlap avoidance — push labels away from other blobs AND other labels
  const labeled = blobs.filter(b => b.partKey && b.labelOffX !== undefined);
  const labelHalfW = (LABEL_EST_W / 2) / sx; // half-size in canvas coords
  const labelHalfH = (LABEL_EST_H / 2) / sy;
  const blobPad = 8;

  for (let iter = 0; iter < 15; iter++) {
    let moved = false;

    // Push labels away from OTHER blobs (most important)
    for (const a of labeled) {
      const acx = (a.minX + a.maxX) / 2 + a.labelOffX;
      const acy = (a.minY + a.maxY) / 2 + a.labelOffY;

      for (const b of blobs) {
        if (b === a) continue; // overlapping own blob is OK
        const bx1 = b.minX - blobPad, by1 = b.minY - blobPad;
        const bx2 = b.maxX + blobPad, by2 = b.maxY + blobPad;
        const lx1 = acx - labelHalfW, ly1 = acy - labelHalfH;
        const lx2 = acx + labelHalfW, ly2 = acy + labelHalfH;

        if (lx1 < bx2 && lx2 > bx1 && ly1 < by2 && ly2 > by1) {
          // Push label away from blob center
          const bcx = (b.minX + b.maxX) / 2;
          const bcy = (b.minY + b.maxY) / 2;
          let dx = acx - bcx || 0.1;
          let dy = acy - bcy || -1;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const push = Math.min(
            Math.min(lx2 - bx1, bx2 - lx1),
            Math.min(ly2 - by1, by2 - ly1)
          ) + 10;
          a.labelOffX += (dx / dist) * push;
          a.labelOffY += (dy / dist) * push;
          moved = true;
        }
      }
    }

    // Push labels away from each other
    for (let i = 0; i < labeled.length; i++) {
      for (let j = i + 1; j < labeled.length; j++) {
        const a = labeled[i], b = labeled[j];
        const ax = ((a.minX + a.maxX) / 2 + a.labelOffX) * sx;
        const ay = ((a.minY + a.maxY) / 2 + a.labelOffY) * sy;
        const bx = ((b.minX + b.maxX) / 2 + b.labelOffX) * sx;
        const by = ((b.minY + b.maxY) / 2 + b.labelOffY) * sy;

        const overlapX = LABEL_EST_W - Math.abs(ax - bx);
        const overlapY = LABEL_EST_H - Math.abs(ay - by);

        if (overlapX > 0 && overlapY > 0) {
          if (overlapX < overlapY) {
            const push = (overlapX / 2 + 4) / sx;
            if (ax <= bx) { a.labelOffX -= push; b.labelOffX += push; }
            else { a.labelOffX += push; b.labelOffX -= push; }
          } else {
            const push = (overlapY / 2 + 4) / sy;
            if (ay <= by) { a.labelOffY -= push; b.labelOffY += push; }
            else { a.labelOffY += push; b.labelOffY -= push; }
          }
          moved = true;
        }
      }
    }

    if (!moved) break;
  }
}

function clearFloatingLabels() {
  document.getElementById('scan-photo-wrap').querySelectorAll('.blob-label').forEach(el => el.remove());
}

let labelDragInfo = null;

function updateFloatingLabels() {
  const wrap = document.getElementById('scan-photo-wrap');
  clearFloatingLabels();

  for (let i = 0; i < blobs.length; i++) {
    const blob = blobs[i];
    if (!blob.partKey || blob.labelOffX === undefined) continue;

    const cx = (blob.minX + blob.maxX) / 2;
    const cy = (blob.minY + blob.maxY) / 2;
    const pos = canvasToWrapCSS(cx + blob.labelOffX, cy + blob.labelOffY);

    const el = document.createElement('div');
    el.className = 'blob-label';
    el.style.left = pos.x + 'px';
    el.style.top = pos.y + 'px';
    el.dataset.blobIdx = i;

    const img = document.createElement('img');
    img.src = getPartImgUrl(blob.partKey);
    img.alt = '';
    el.appendChild(img);

    const span = document.createElement('span');
    span.textContent = blob.partName;
    el.appendChild(span);

    wrap.appendChild(el);

    // Drag + tap handling
    el.addEventListener('pointerdown', labelPointerDown);
    el.addEventListener('pointermove', labelPointerMove);
    el.addEventListener('pointerup', labelPointerUp);
  }
}

function labelPointerDown(e) {
  e.stopPropagation();
  e.preventDefault();
  const el = e.currentTarget;
  el.setPointerCapture(e.pointerId);
  labelDragInfo = {
    el,
    blobIdx: parseInt(el.dataset.blobIdx),
    startX: e.clientX, startY: e.clientY,
    lastX: e.clientX, lastY: e.clientY,
    moved: false,
  };
}

function labelPointerMove(e) {
  if (!labelDragInfo) return;
  e.preventDefault();
  const dx = e.clientX - labelDragInfo.lastX;
  const dy = e.clientY - labelDragInfo.lastY;

  if (Math.abs(e.clientX - labelDragInfo.startX) > 5 ||
      Math.abs(e.clientY - labelDragInfo.startY) > 5) {
    labelDragInfo.moved = true;
  }
  if (!labelDragInfo.moved) return;

  labelDragInfo.lastX = e.clientX;
  labelDragInfo.lastY = e.clientY;
  labelDragInfo.el.classList.add('dragging');

  const blob = blobs[labelDragInfo.blobIdx];
  const canvas = document.getElementById('scan-canvas');
  const cRect = canvas.getBoundingClientRect();
  blob.labelOffX += dx * (canvas.width / cRect.width);
  blob.labelOffY += dy * (canvas.height / cRect.height);

  const cx = (blob.minX + blob.maxX) / 2;
  const cy = (blob.minY + blob.maxY) / 2;
  const pos = canvasToWrapCSS(cx + blob.labelOffX, cy + blob.labelOffY);
  labelDragInfo.el.style.left = pos.x + 'px';
  labelDragInfo.el.style.top = pos.y + 'px';

  drawAnnotatedPhoto();
}

function labelPointerUp(e) {
  if (!labelDragInfo) return;
  const info = labelDragInfo;
  labelDragInfo = null;
  info.el.classList.remove('dragging');

  if (!info.moved) {
    // Tap on label — select blob and open picker
    selectedBlobIdx = info.blobIdx;
    drawAnnotatedPhoto();
    updateFloatingLabels();
    showPartPicker(blobs[info.blobIdx]);
  }
}

// ── Event Listeners ──

document.getElementById('scan-btn').addEventListener('click', openScan);
document.getElementById('scan-close').addEventListener('click', closeScan);

document.getElementById('scan-all-same').addEventListener('click', () => {
  if (blobs.length === 0) return;
  pickerApplyAll = true;
  // Use first unlabeled blob for color reference, fall back to first blob
  const refBlob = blobs.find((b) => !b.partKey) || blobs[0];
  selectedBlobIdx = blobs.indexOf(refBlob);
  drawAnnotatedPhoto();
  showPartPicker(refBlob);
});

document.getElementById('scan-guess-all').addEventListener('click', async () => {
  if (blobs.length === 0) return;
  const setNum = window.lostbricks.getCurrentSetNum();
  const parts = window.lostbricks.getCurrentSetParts();
  const partMap = new Map();
  for (const p of parts) partMap.set(`${p.partNum}_${p.color.id}`, p.name);

  let guessed = 0;
  for (let i = 0; i < blobs.length; i++) {
    const blob = blobs[i];
    if (blob.partKey) continue; // skip already labeled
    const jpegB64 = blobToBase64(blob);
    const result = await apiSuggest(setNum, jpegB64);
    if (result.suggestions.length > 0) {
      const top = result.suggestions[0];
      blob.partKey = top.key;
      blob.partName = partMap.get(top.key) || top.key;
      blob.autoAssignedKey = top.key;
      // Send to learning server
      const [partNum, colorId] = top.key.split('_');
      apiLabel(setNum, partNum, colorId, jpegB64, blob.avgR, blob.avgG, blob.avgB, blob.histR, blob.histG, blob.histB);
      guessed++;
    }
  }
  if (guessed) {
    initLabelPositions();
    updateStatus();
    drawAnnotatedPhoto();
    updateFloatingLabels();
    console.log(`guessed ${guessed} blobs`);
  }
});

document.getElementById('scan-clear-all').addEventListener('click', () => {
  for (const blob of blobs) {
    blob.partKey = null;
    blob.partName = null;
    blob.autoAssignedKey = null;
    blob.labelOffX = undefined;
    blob.labelOffY = undefined;
  }
  clearFloatingLabels();
  updateStatus();
  drawAnnotatedPhoto();
});

// ── Crop Phase ──

let rawCanvas = null; // full uncropped frame
let cropRect = { x: 0, y: 0, w: 0, h: 0 }; // in image coords
let cropDrag = null; // { edge, startPos, startRect }

function showCrop(canvas) {
  rawCanvas = canvas;
  stopCamera();
  // Default crop: 5% inset
  const inset = 0.05;
  cropRect = {
    x: Math.round(canvas.width * inset),
    y: Math.round(canvas.height * inset),
    w: Math.round(canvas.width * (1 - 2 * inset)),
    h: Math.round(canvas.height * (1 - 2 * inset)),
  };
  document.getElementById('scan-viewfinder').classList.add('hidden');
  document.getElementById('scan-crop').classList.remove('hidden');
  drawCrop();
}

function drawCrop() {
  const canvas = document.getElementById('crop-canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = rawCanvas.width;
  canvas.height = rawCanvas.height;

  // Draw dimmed full image
  ctx.drawImage(rawCanvas, 0, 0);
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Draw bright crop region
  const { x, y, w, h } = cropRect;
  ctx.drawImage(rawCanvas, x, y, w, h, x, y, w, h);

  // Draw crop border
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth = 4;
  ctx.strokeRect(x, y, w, h);

  // Draw corner handles
  const hs = 20;
  ctx.fillStyle = '#e94560';
  for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
    ctx.fillRect(cx - hs / 2, cy - hs / 2, hs, hs);
  }
}

function getCropEdge(px, py) {
  const { x, y, w, h } = cropRect;
  const margin = 40; // touch target size in image coords
  const canvas = document.getElementById('crop-canvas');
  const rect = canvas.getBoundingClientRect();
  const scale = canvas.width / rect.width;
  const m = margin * scale;

  // Check corners first
  if (Math.abs(px - x) < m && Math.abs(py - y) < m) return 'tl';
  if (Math.abs(px - (x + w)) < m && Math.abs(py - y) < m) return 'tr';
  if (Math.abs(px - x) < m && Math.abs(py - (y + h)) < m) return 'bl';
  if (Math.abs(px - (x + w)) < m && Math.abs(py - (y + h)) < m) return 'br';
  // Check edges
  if (Math.abs(px - x) < m && py > y && py < y + h) return 'l';
  if (Math.abs(px - (x + w)) < m && py > y && py < y + h) return 'r';
  if (Math.abs(py - y) < m && px > x && px < x + w) return 't';
  if (Math.abs(py - (y + h)) < m && px > x && px < x + w) return 'b';
  // Inside = move whole rect
  if (px > x && px < x + w && py > y && py < y + h) return 'move';
  return null;
}

function getCropCanvasPos(e) {
  const canvas = document.getElementById('crop-canvas');
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  let cx, cy;
  if (e.touches) { cx = e.touches[0].clientX; cy = e.touches[0].clientY; }
  else { cx = e.clientX; cy = e.clientY; }
  return { x: (cx - rect.left) * scaleX, y: (cy - rect.top) * scaleY };
}

const cropCanvas = document.getElementById('crop-canvas');

cropCanvas.addEventListener('pointerdown', (e) => {
  e.preventDefault();
  const pos = getCropCanvasPos(e);
  const edge = getCropEdge(pos.x, pos.y);
  if (!edge) return;
  cropDrag = { edge, startX: pos.x, startY: pos.y, startRect: { ...cropRect } };
});

cropCanvas.addEventListener('pointermove', (e) => {
  if (!cropDrag) return;
  e.preventDefault();
  const pos = getCropCanvasPos(e);
  const dx = pos.x - cropDrag.startX;
  const dy = pos.y - cropDrag.startY;
  const s = cropDrag.startRect;
  const minSize = 100;

  switch (cropDrag.edge) {
    case 'move':
      cropRect.x = Math.max(0, Math.min(rawCanvas.width - s.w, s.x + dx));
      cropRect.y = Math.max(0, Math.min(rawCanvas.height - s.h, s.y + dy));
      break;
    case 'tl':
      cropRect.x = Math.min(s.x + dx, s.x + s.w - minSize);
      cropRect.y = Math.min(s.y + dy, s.y + s.h - minSize);
      cropRect.w = s.w - (cropRect.x - s.x);
      cropRect.h = s.h - (cropRect.y - s.y);
      break;
    case 'tr':
      cropRect.w = Math.max(minSize, s.w + dx);
      cropRect.y = Math.min(s.y + dy, s.y + s.h - minSize);
      cropRect.h = s.h - (cropRect.y - s.y);
      break;
    case 'bl':
      cropRect.x = Math.min(s.x + dx, s.x + s.w - minSize);
      cropRect.w = s.w - (cropRect.x - s.x);
      cropRect.h = Math.max(minSize, s.h + dy);
      break;
    case 'br':
      cropRect.w = Math.max(minSize, s.w + dx);
      cropRect.h = Math.max(minSize, s.h + dy);
      break;
    case 'l':
      cropRect.x = Math.min(s.x + dx, s.x + s.w - minSize);
      cropRect.w = s.w - (cropRect.x - s.x);
      break;
    case 'r':
      cropRect.w = Math.max(minSize, s.w + dx);
      break;
    case 't':
      cropRect.y = Math.min(s.y + dy, s.y + s.h - minSize);
      cropRect.h = s.h - (cropRect.y - s.y);
      break;
    case 'b':
      cropRect.h = Math.max(minSize, s.h + dy);
      break;
  }

  // Clamp to canvas bounds
  cropRect.x = Math.max(0, cropRect.x);
  cropRect.y = Math.max(0, cropRect.y);
  cropRect.w = Math.min(cropRect.w, rawCanvas.width - cropRect.x);
  cropRect.h = Math.min(cropRect.h, rawCanvas.height - cropRect.y);

  drawCrop();
});

cropCanvas.addEventListener('pointerup', () => { cropDrag = null; });

document.getElementById('crop-confirm').addEventListener('click', () => {
  // Crop the raw canvas
  const { x, y, w, h } = cropRect;
  capturedCanvas = document.createElement('canvas');
  capturedCanvas.width = Math.round(w);
  capturedCanvas.height = Math.round(h);
  capturedCanvas.getContext('2d').drawImage(rawCanvas, Math.round(x), Math.round(y), Math.round(w), Math.round(h), 0, 0, Math.round(w), Math.round(h));

  const detected = segmentImage(capturedCanvas);
  document.getElementById('scan-crop').classList.add('hidden');
  showAnnotated(detected);
});

document.getElementById('crop-retake').addEventListener('click', () => {
  document.getElementById('scan-crop').classList.add('hidden');
  document.getElementById('scan-viewfinder').classList.remove('hidden');
  startCamera();
});

// ── Capture & Navigation ──

document.getElementById('scan-capture').addEventListener('click', () => {
  showCrop(captureFrame());
});

document.getElementById('scan-retake').addEventListener('click', () => {
  document.getElementById('scan-annotate').classList.add('hidden');
  document.getElementById('scan-viewfinder').classList.remove('hidden');
  document.getElementById('part-picker').classList.add('hidden');
  blobs = [];
  selectedBlobIdx = null;
  clearFloatingLabels();
  startCamera();
});

document.getElementById('scan-done').addEventListener('click', () => {
  // Apply all labeled blob counts in batch
  for (const blob of blobs) {
    if (blob.partKey) {
      window.lostbricks.incrementPart(blob.partKey);
    }
  }
  // Save full session data (fire-and-forget), then close
  apiSaveSession();
  closeScan();
});
document.getElementById('picker-close').addEventListener('click', () => {
  hidePartPicker();
  selectedBlobIdx = null;
  drawAnnotatedPhoto();
  updateFloatingLabels();
});

document.getElementById('picker-delete').addEventListener('click', () => {
  if (selectedBlobIdx !== null) {
    deletedBlobs.push(blobs[selectedBlobIdx]);
    blobs.splice(selectedBlobIdx, 1);
    selectedBlobIdx = null;
    hidePartPicker();
    updateStatus();
    drawAnnotatedPhoto();
    updateFloatingLabels();
  }
});
