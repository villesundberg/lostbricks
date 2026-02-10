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

  const mask = new Uint8Array(totalPixels);
  for (let i = 0; i < totalPixels; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
    mask[i] = brightness < 200 ? 1 : 0;
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
      totalR += data[idx * 4];
      totalG += data[idx * 4 + 1];
      totalB += data[idx * 4 + 2];

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
    });
  }

  // Filter: min 0.2% of image, max 25% (reject background)
  const minArea = totalPixels * 0.002;
  const maxArea = totalPixels * 0.25;
  return components
    .filter((c) => c.pixelCount >= minArea && c.pixelCount <= maxArea)
    .sort((a, b) => b.pixelCount - a.pixelCount);
}

// Color distance (simple Euclidean in RGB)
function colorDist(r1, g1, b1, r2, g2, b2) {
  return Math.sqrt((r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2);
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

async function apiLabel(setNum, partNum, colorId, jpegB64) {
  try {
    const res = await fetch('/api/label', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setNum, partNum, colorId: String(colorId), jpeg: jpegB64 }),
    });
    const data = await res.json();
    if (data.ok) console.log(`learn: ${partNum}_${colorId} (${data.count} crops for set)`);
    else console.warn('learn error:', data.error);
  } catch (e) {
    console.warn('learn send failed:', e);
  }
}

async function apiSuggest(setNum, jpegB64) {
  try {
    const res = await fetch('/api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ setNum, jpeg: jpegB64 }),
    });
    const data = await res.json();
    return data.suggestions || []; // [{ key, dist }, ...]
  } catch (e) {
    console.warn('suggest failed:', e);
    return [];
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

let blobs = []; // { minX, minY, maxX, maxY, avgR, avgG, avgB, partKey, partName }
let capturedCanvas = null; // original captured frame
let selectedBlobIdx = null;

function openScan() {
  document.getElementById('scan-modal').classList.remove('hidden');
  document.getElementById('scan-viewfinder').classList.remove('hidden');
  document.getElementById('scan-crop').classList.add('hidden');
  document.getElementById('scan-annotate').classList.add('hidden');
  document.getElementById('part-picker').classList.add('hidden');
  blobs = [];
  selectedBlobIdx = null;
  capturedCanvas = null;
  startCamera();
}

function closeScan() {
  stopCamera();
  document.getElementById('scan-modal').classList.add('hidden');
}

function showAnnotated(detected) {
  blobs = detected.map((c) => ({
    ...c, partKey: null, partName: null,
  }));
  stopCamera();
  document.getElementById('scan-viewfinder').classList.add('hidden');
  document.getElementById('scan-annotate').classList.remove('hidden');
  updateStatus();
  drawAnnotatedPhoto();
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
      // Label text
      ctx.font = 'bold 24px sans-serif';
      ctx.fillStyle = '#2ecc71';
      ctx.fillRect(x, y - 28, Math.min(ctx.measureText(blob.partName).width + 8, w), 28);
      ctx.fillStyle = '#000';
      ctx.fillText(blob.partName, x + 4, y - 6, w - 8);
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

  // Sample average color from foreground pixels only
  const ctx = capturedCanvas.getContext('2d');
  const imageData = ctx.getImageData(Math.round(x1), Math.round(y1), Math.round(w), Math.round(h));
  const data = imageData.data;
  const totalPx = data.length / 4;
  let totalR = 0, totalG = 0, totalB = 0, fgCount = 0;
  for (let i = 0; i < totalPx; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
    if (brightness < 200) {
      totalR += r;
      totalG += g;
      totalB += b;
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
    partKey: null, partName: null,
  };
  blobs.push(newBlob);
  selectedBlobIdx = blobs.length - 1;
  updateStatus();
  drawAnnotatedPhoto();
  showPartPicker(newBlob);
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

function showPartPicker(blob) {
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

  // Score by color distance + remaining need
  if (blob) {
    const { avgR, avgG, avgB } = blob;
    pickerParts = pickerParts.map((p) => {
      const key = `${p.partNum}_${p.color.id}`;
      const [pr, pg, pb] = hexToRGB(p.color.rgb);
      const dist = colorDist(avgR, avgG, avgB, pr, pg, pb);
      const have = window.lostbricks.getHave(key);
      const remaining = Math.max(0, p.qty - have);
      const score = dist - remaining * 2;
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
    apiSuggest(setNum, jpegB64).then((suggestions) => {
      if (suggestions.length === 0) return;
      // Apply ML boost to matching parts
      const boostMap = new Map();
      suggestions.forEach((s, i) => {
        // Top suggestion gets biggest boost, decaying
        boostMap.set(s.key, 100 - i * 8);
      });
      for (const p of pickerParts) {
        const key = `${p.partNum}_${p.color.id}`;
        const boost = boostMap.get(key);
        if (boost) {
          p.mlBoost = boost;
          p.score -= boost; // lower score = better
        }
      }
      pickerParts.sort((a, b) => a.score - b.score);
      // Re-render with updated rankings
      buildColorChips();
      renderPickerGrid();
      console.log(`ML suggest: ${suggestions.length} matches, top=${suggestions[0].key}`);
    });
  }

  document.getElementById('picker-title').textContent = 'Pick a part';
  document.getElementById('part-picker').classList.remove('hidden');
}

function renderPickerGrid() {
  const grid = document.getElementById('picker-grid');
  let filtered = pickerParts;
  if (pickerColorFilter !== null) {
    filtered = pickerParts.filter((p) => p.color.id === pickerColorFilter);
  }

  grid.innerHTML = filtered.map((p) => {
    const key = `${p.partNum}_${p.color.id}`;
    const remaining = p.remaining !== undefined ? p.remaining : '';
    return `
      <div class="picker-item" data-key="${key}" data-name="${p.name}">
        <img src="${p.imgUrl || ''}" alt="" loading="lazy">
        <div class="picker-label">${p.name}${remaining ? ` (${remaining})` : ''}</div>
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
  if (selectedBlobIdx === null) return;

  const blob = blobs[selectedBlobIdx];
  blob.partKey = key;
  blob.partName = name;

  window.lostbricks.incrementPart(key);

  // Store crop locally + send to learning server
  const [partNum, colorId] = key.split('_');
  const jpegB64 = blobToBase64(blob);

  // Send to server (fire-and-forget)
  apiLabel(window.lostbricks.getCurrentSetNum(), partNum, colorId, jpegB64);

  // Also store in IndexedDB
  try {
    const jpeg = await fetch(`data:image/jpeg;base64,${jpegB64}`).then((r) => r.blob());
    await storeCrop(window.lostbricks.getCurrentSetNum(), partNum, parseInt(colorId), jpeg);
  } catch (e) {
    console.warn('Failed to store crop:', e);
  }

  hidePartPicker();
  selectedBlobIdx = null;
  updateStatus();
  drawAnnotatedPhoto();
}

// ── Event Listeners ──

document.getElementById('scan-btn').addEventListener('click', openScan);
document.getElementById('scan-close').addEventListener('click', closeScan);

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
  startCamera();
});

document.getElementById('scan-done').addEventListener('click', closeScan);
document.getElementById('picker-close').addEventListener('click', () => {
  hidePartPicker();
  selectedBlobIdx = null;
  drawAnnotatedPhoto();
});

document.getElementById('picker-delete').addEventListener('click', () => {
  if (selectedBlobIdx !== null) {
    blobs.splice(selectedBlobIdx, 1);
    selectedBlobIdx = null;
    hidePartPicker();
    updateStatus();
    drawAnnotatedPhoto();
  }
});
