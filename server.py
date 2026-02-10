#!/usr/bin/env python3
"""LostBricks server — static files + learning API."""

import http.server
import ssl
import sys
import json
import os
import time
import io
import threading
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Config ──

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
NO_TLS = '--no-tls' in sys.argv
CROP_DIR = Path(__file__).parent / 'data' / 'crops'
CATALOG_DIR = Path(__file__).parent / 'data' / 'catalog'
STATE_FILE = Path(__file__).parent / 'data' / 'state.json'
FLAGS_FILE = Path(__file__).parent / 'data' / 'flags.json'
SESSION_DIR = Path(__file__).parent / 'data' / 'sessions'
SAT_THRESHOLD = 0.15  # saturation threshold for foreground recovery

# ── In-memory index ──
# { setNum: [ { key: "partNum_colorId", vec: np.array, path: str }, ... ] }
index = {}
index_lock = threading.Lock()

# ── Learned color map ──
# { setNum: { colorId: { r, g, b, count } } }
# Tracks actual photographed RGB per LEGO color (running average)
learned_colors = {}
learned_threshold = {}  # { setNum: { threshold: int, sampleCount: int } }
LEARNED_FILE = Path(__file__).parent / 'data' / 'learned.json'


# ── CNN Feature Extractor ──

cnn_model = None
cnn_preprocess = None

def load_cnn():
    """Load MobileNetV3-Small as feature extractor (576-dim embeddings)."""
    global cnn_model, cnn_preprocess
    print('Loading MobileNetV3-Small...')
    model = models.mobilenet_v3_small(weights='DEFAULT')
    model.classifier = torch.nn.Identity()  # remove classification head → 576-dim
    model.eval()
    cnn_model = model
    cnn_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print('  MobileNetV3-Small ready (576-dim embeddings)')


def img_to_vec(img_bytes):
    """JPEG bytes → normalized 576-dim CNN embedding."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tensor = cnn_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = cnn_model(tensor)
    vec = features.squeeze().numpy()
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def load_or_compute_vec(filepath):
    """Load cached embedding or compute + cache it."""
    cache_path = Path(str(filepath) + '.vec.npy')
    if cache_path.exists():
        return np.load(cache_path)
    vec = img_to_vec(Path(filepath).read_bytes())
    np.save(cache_path, vec)
    return vec


# ── Color Classifier ──

color_clf = None  # sklearn Pipeline (scaler + SVM)
color_train_X = []  # list of 48-dim histogram arrays
color_train_y = []  # list of color ID strings


def crop_to_histogram(img_bytes, threshold=200):
    """Compute 48-dim normalized histogram from crop foreground pixels."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    pixels = arr.reshape(-1, 3)

    brightness = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
    mx = np.max(pixels, axis=1)
    mn = np.min(pixels, axis=1)
    sat = np.where(mx > 0, (mx - mn) / mx, 0)
    fg_mask = (brightness < threshold) | (sat > SAT_THRESHOLD)
    fg = pixels[fg_mask]

    hist = np.zeros(48, dtype=np.float64)
    if len(fg) == 0:
        return hist

    for i in range(3):
        bins = np.clip(fg[:, i].astype(int) >> 4, 0, 15)
        counts = np.bincount(bins, minlength=16)[:16]
        hist[i * 16:i * 16 + 16] = counts

    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def train_color_classifier():
    """Train SVM on accumulated histogram data."""
    global color_clf
    if len(color_train_X) < 5 or len(set(color_train_y)) < 2:
        return
    X = np.array(color_train_X)
    y = np.array(color_train_y)
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, C=10, gamma='scale')),
    ])
    clf.fit(X, y)
    color_clf = clf
    acc = clf.score(X, y)
    print(f'  color classifier: {len(X)} samples, {len(set(y))} classes, train acc={acc:.1%}')


def predict_color(hist_48):
    """Return {colorId: probability} for a 48-dim histogram."""
    if color_clf is None:
        return {}
    X = np.array([hist_48])
    probs = color_clf.predict_proba(X)[0]
    classes = color_clf.classes_
    return {str(c): round(float(p), 4) for c, p in zip(classes, probs)}


def build_color_training_data():
    """Build per-crop histograms from existing crop images for classifier training."""
    global color_train_X, color_train_y
    color_train_X = []
    color_train_y = []
    for set_num, entries in index.items():
        threshold = learned_threshold.get(set_num, {}).get('threshold', 200)
        for e in entries:
            try:
                hist = crop_to_histogram(Path(e['path']).read_bytes(), threshold)
                color_id = e['key'].split('_')[1]
                color_train_X.append(hist)
                color_train_y.append(color_id)
            except Exception:
                pass
    print(f'  color training data: {len(color_train_X)} samples')


def load_index():
    """Scan data/crops + data/catalog on startup, build in-memory index with CNN embeddings."""
    for base_dir in [CROP_DIR, CATALOG_DIR]:
        if not base_dir.exists():
            continue
        is_catalog = (base_dir == CATALOG_DIR)
        label = 'catalog' if is_catalog else 'crops'
        for set_dir in base_dir.iterdir():
            if not set_dir.is_dir():
                continue
            set_num = set_dir.name
            if set_num not in index:
                index[set_num] = []
            cached = 0
            computed = 0
            for f in sorted(set_dir.glob('*.jpg')):
                # crops: {partNum}_{colorId}_{timestamp}.jpg
                # catalog: {partNum}_{colorId}.jpg
                parts = f.stem.split('_')
                if len(parts) < 2:
                    continue
                key = f'{parts[0]}_{parts[1]}'
                try:
                    cache_path = Path(str(f) + '.vec.npy')
                    if cache_path.exists():
                        vec = np.load(cache_path)
                        cached += 1
                    else:
                        vec = img_to_vec(f.read_bytes())
                        np.save(cache_path, vec)
                        computed += 1
                    index[set_num].append({'key': key, 'vec': vec, 'path': str(f), 'catalog': is_catalog})
                except Exception as e:
                    print(f'  skip {f}: {e}')
            count = cached + computed
            if count:
                print(f'  {set_num}: {count} {label} ({cached} cached, {computed} computed)')


def load_learned():
    """Load learned data from disk."""
    global learned_colors, learned_threshold
    if LEARNED_FILE.exists():
        data = json.loads(LEARNED_FILE.read_text())
        if isinstance(data, dict) and 'colors' in data:
            learned_colors = data.get('colors', {})
            learned_threshold = data.get('thresholds', {})
        else:
            # Legacy format: just colors
            learned_colors = data
        total = sum(sum(c['count'] for c in s.values()) for s in learned_colors.values())
        thresholds = {k: v['threshold'] for k, v in learned_threshold.items()}
        print(f'  loaded learned: {total} color samples, thresholds={thresholds}')


def save_learned():
    """Persist all learned data to disk."""
    LEARNED_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEARNED_FILE.write_text(json.dumps({
        'colors': learned_colors,
        'thresholds': learned_threshold,
    }))


def update_learned_color(set_num, color_id, avg_r, avg_g, avg_b, hist_r=None, hist_g=None, hist_b=None):
    """Update running average of photographed color and accumulated histograms."""
    if set_num not in learned_colors:
        learned_colors[set_num] = {}
    colors = learned_colors[set_num]
    cid = str(color_id)
    if cid not in colors:
        colors[cid] = {'r': avg_r, 'g': avg_g, 'b': avg_b, 'count': 1}
        if hist_r:
            colors[cid]['histR'] = list(hist_r)
            colors[cid]['histG'] = list(hist_g)
            colors[cid]['histB'] = list(hist_b)
    else:
        c = colors[cid]
        n = c['count']
        # Running average
        c['r'] = round((c['r'] * n + avg_r) / (n + 1), 1)
        c['g'] = round((c['g'] * n + avg_g) / (n + 1), 1)
        c['b'] = round((c['b'] * n + avg_b) / (n + 1), 1)
        c['count'] = n + 1
        # Accumulate histograms (sum of all samples — normalize when comparing)
        if hist_r:
            if 'histR' not in c:
                c['histR'] = list(hist_r)
                c['histG'] = list(hist_g)
                c['histB'] = list(hist_b)
            else:
                for i in range(len(hist_r)):
                    c['histR'][i] = round(c['histR'][i] + hist_r[i], 1)
                    c['histG'][i] = round(c['histG'][i] + hist_g[i], 1)
                    c['histB'][i] = round(c['histB'][i] + hist_b[i], 1)
    save_learned()


def compute_threshold(cropped_path, blobs):
    """Analyze a cropped photo + labeled bounding boxes to find optimal brightness threshold.
    Returns the brightness value that best separates part pixels from background."""
    try:
        img = Image.open(cropped_path).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]

        # Compute brightness for all pixels
        brightness = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2])

        # Build mask of labeled blob regions
        part_mask = np.zeros((h, w), dtype=bool)
        for b in blobs:
            if not b.get('partKey'):
                continue
            y1 = max(0, int(b['minY']))
            y2 = min(h, int(b['maxY']))
            x1 = max(0, int(b['minX']))
            x2 = min(w, int(b['maxX']))
            part_mask[y1:y2, x1:x2] = True

        part_brightness = brightness[part_mask]
        bg_brightness = brightness[~part_mask]

        if len(part_brightness) == 0 or len(bg_brightness) == 0:
            return None

        # Find threshold that best separates: maximize gap between
        # "fraction of bg above threshold" and "fraction of parts above threshold"
        # Try thresholds from 100 to 240
        best_t = 200
        best_score = -1
        for t in range(100, 241, 5):
            bg_above = np.mean(bg_brightness > t)  # want high (bg is bright)
            part_below = np.mean(part_brightness < t)  # want high (parts are dark)
            score = bg_above + part_below
            if score > best_score:
                best_score = score
                best_t = t

        return best_t
    except Exception as e:
        print(f'  threshold compute error: {e}')
        return None


def update_learned_threshold(set_num, new_threshold):
    """Update running average of learned threshold for a set."""
    if set_num not in learned_threshold:
        learned_threshold[set_num] = {'threshold': new_threshold, 'sampleCount': 1}
    else:
        lt = learned_threshold[set_num]
        n = lt['sampleCount']
        lt['threshold'] = round((lt['threshold'] * n + new_threshold) / (n + 1))
        lt['sampleCount'] = n + 1
    save_learned()


def rebuild_histograms():
    """Bootstrap histograms from existing crop images for all sets."""
    for set_num, entries in index.items():
        threshold = learned_threshold.get(set_num, {}).get('threshold', 200)
        colors = learned_colors.get(set_num, {})

        # Clear existing histograms (rebuild from scratch)
        for c in colors.values():
            c.pop('histR', None)
            c.pop('histG', None)
            c.pop('histB', None)

        count = 0
        for e in entries:
            try:
                img = Image.open(e['path']).convert('RGB')
                arr = np.array(img)
                pixels = arr.reshape(-1, 3)
                # Brightness threshold + saturation recovery to get foreground
                brightness = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
                mx = np.max(pixels, axis=1).astype(np.float32)
                mn = np.min(pixels, axis=1).astype(np.float32)
                sat = np.where(mx > 0, (mx - mn) / mx, 0)
                fg_mask = (brightness < threshold) | (sat > 0.15)
                fg = pixels[fg_mask]
                if len(fg) == 0:
                    continue

                hist_r = [0.0] * 16
                hist_g = [0.0] * 16
                hist_b = [0.0] * 16
                for px in fg:
                    hist_r[min(15, int(px[0]) >> 4)] += 1
                    hist_g[min(15, int(px[1]) >> 4)] += 1
                    hist_b[min(15, int(px[2]) >> 4)] += 1

                color_id = e['key'].split('_')[1]
                cid = str(color_id)
                if cid in colors:
                    c = colors[cid]
                    if 'histR' not in c:
                        c['histR'] = hist_r
                        c['histG'] = hist_g
                        c['histB'] = hist_b
                    else:
                        for i in range(16):
                            c['histR'][i] += hist_r[i]
                            c['histG'][i] += hist_g[i]
                            c['histB'][i] += hist_b[i]
                    count += 1
            except Exception:
                pass

        if count:
            print(f'  rebuilt histograms for {set_num}: {count} crops')
    save_learned()


def add_to_index(set_num, key, vec, path):
    """Add a new entry to the in-memory index."""
    with index_lock:
        if set_num not in index:
            index[set_num] = []
        index[set_num].append({'key': key, 'vec': vec, 'path': path})


def suggest(set_num, query_vec, top_k=50):
    """Return top-K nearest parts for a query vector within a set."""
    with index_lock:
        entries = index.get(set_num, [])
    if not entries:
        return []

    # Stack all vectors, compute L2 distances
    vecs = np.stack([e['vec'] for e in entries])
    dists = np.linalg.norm(vecs - query_vec, axis=1)
    sorted_idx = np.argsort(dists)

    # Deduplicate by key, keep best distance per part
    seen = {}
    results = []
    for i in sorted_idx:
        key = entries[i]['key']
        d = float(dists[i])
        if key not in seen:
            seen[key] = True
            results.append({'key': key, 'dist': round(d, 4)})
            if len(results) >= top_k:
                break
    return results


# ── HTTP Handler ──

class ReuseHTTPServer(http.server.HTTPServer):
    allow_reuse_address = True

PREFIX = '/lostbricks'

class Handler(http.server.SimpleHTTPRequestHandler):
    def _strip_prefix(self):
        """Strip /lostbricks prefix; redirect bare /lostbricks to /lostbricks/."""
        if self.path == PREFIX:
            self.send_response(301)
            self.send_header('Location', PREFIX + '/')
            self.end_headers()
            return False
        if self.path.startswith(PREFIX + '/'):
            self.path = self.path[len(PREFIX):] or '/'
        return True

    def do_GET(self):
        if not self._strip_prefix():
            return
        if self.path == '/api/state':
            self.handle_get_state()
        elif self.path.startswith('/api/learned/'):
            self.handle_get_learned()
        elif self.path.startswith('/api/debug/'):
            self.handle_debug()
        else:
            super().do_GET()

    def do_POST(self):
        if not self._strip_prefix():
            return
        if self.path == '/api/label':
            self.handle_label()
        elif self.path == '/api/suggest':
            self.handle_suggest()
        elif self.path == '/api/state':
            self.handle_save_state()
        elif self.path == '/api/session':
            self.handle_save_session()
        elif self.path == '/api/reject':
            self.handle_reject()
        elif self.path == '/api/flag':
            self.handle_flag()
        elif self.path == '/api/dedup':
            self.handle_dedup()
        elif self.path == '/api/reset':
            self.handle_reset()
        elif self.path == '/api/relabel':
            self.handle_relabel()
        elif self.path == '/api/delete-crop':
            self.handle_delete_crop()
        elif self.path == '/api/bootstrap':
            self.handle_bootstrap()
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        # CORS preflight
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def read_body(self):
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length)

    def json_response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def handle_label(self):
        """Receive a labeled crop: { setNum, partNum, colorId, jpeg: base64 }"""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']
            part_num = data['partNum']
            color_id = data['colorId']
            jpeg_b64 = data['jpeg']

            # Decode JPEG
            import base64
            jpeg_bytes = base64.b64decode(jpeg_b64)

            # Compute feature vector
            vec = img_to_vec(jpeg_bytes)

            # Save to disk
            set_dir = CROP_DIR / set_num
            set_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time() * 1000)
            filename = f'{part_num}_{color_id}_{ts}.jpg'
            filepath = set_dir / filename
            filepath.write_bytes(jpeg_bytes)

            # Cache the CNN embedding
            np.save(str(filepath) + '.vec.npy', vec)

            # Add to index
            key = f'{part_num}_{color_id}'
            add_to_index(set_num, key, vec, str(filepath))

            # Learn color from blob's average RGB + histograms if provided
            avg_r = data.get('avgR')
            avg_g = data.get('avgG')
            avg_b = data.get('avgB')
            hist_r = data.get('histR')
            hist_g = data.get('histG')
            hist_b = data.get('histB')
            if avg_r is not None and avg_g is not None and avg_b is not None:
                update_learned_color(set_num, color_id, avg_r, avg_g, avg_b, hist_r, hist_g, hist_b)

            # Add to color classifier training data + retrain
            threshold = learned_threshold.get(set_num, {}).get('threshold', 200)
            hist = crop_to_histogram(jpeg_bytes, threshold)
            color_train_X.append(hist)
            color_train_y.append(color_id)
            train_color_classifier()

            count = len(index.get(set_num, []))
            colors = learned_colors.get(set_num, {})
            self.json_response({'ok': True, 'cropCount': count, 'learnedColors': colors})
            print(f'  label: {set_num} {key} ({count} total for set)')

        except Exception as e:
            print(f'  label error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_suggest(self):
        """Suggest parts for a blob crop: { setNum, jpeg: base64 } → ranked keys + color probs."""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']
            jpeg_b64 = data['jpeg']

            import base64
            jpeg_bytes = base64.b64decode(jpeg_b64)
            query_vec = img_to_vec(jpeg_bytes)

            results = suggest(set_num, query_vec)

            # Color prediction from histogram
            threshold = learned_threshold.get(set_num, {}).get('threshold', 200)
            hist = crop_to_histogram(jpeg_bytes, threshold)
            color_probs = predict_color(hist)

            colors = learned_colors.get(set_num, {})
            with index_lock:
                crop_count = len(index.get(set_num, []))
            self.json_response({
                'suggestions': results,
                'colorProbs': color_probs,
                'learnedColors': colors,
                'cropCount': crop_count,
                'threshold': threshold,
            })

        except Exception as e:
            print(f'  suggest error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_get_learned(self):
        """Return learned data for a set: color map, crop count, stats."""
        set_num = self.path.split('/api/learned/')[1]
        with index_lock:
            crop_count = len(index.get(set_num, []))
            # Count unique parts in index
            unique_keys = set(e['key'] for e in index.get(set_num, []))
        colors = learned_colors.get(set_num, {})
        threshold = learned_threshold.get(set_num, {}).get('threshold')
        self.json_response({
            'colors': colors,
            'cropCount': crop_count,
            'uniqueParts': len(unique_keys),
            'threshold': threshold,
        })

    def handle_get_state(self):
        """Return saved app state."""
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            self.json_response(data)
        else:
            self.json_response(None)

    def handle_save_state(self):
        """Save app state to disk."""
        try:
            data = json.loads(self.read_body())
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps(data))
            self.json_response({'ok': True})
            print(f'  state saved ({len(data.get("sets", {}))} sets)')
        except Exception as e:
            print(f'  state save error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_save_session(self):
        """Save a complete scan session: raw photo, cropped photo, blobs, deleted blobs."""
        try:
            import base64
            data = json.loads(self.read_body())
            set_num = data['setNum']
            ts = data.get('timestamp', int(time.time() * 1000))

            session_dir = SESSION_DIR / set_num / str(ts)
            session_dir.mkdir(parents=True, exist_ok=True)

            # Save raw photo
            if data.get('rawPhoto'):
                (session_dir / 'raw.jpg').write_bytes(base64.b64decode(data['rawPhoto']))

            # Save cropped photo
            if data.get('croppedPhoto'):
                (session_dir / 'cropped.jpg').write_bytes(base64.b64decode(data['croppedPhoto']))

            # Save session metadata (everything except the large base64 images)
            meta = {
                'setNum': set_num,
                'timestamp': ts,
                'cropRect': data.get('cropRect'),
                'blobs': data.get('blobs', []),
                'deletedBlobs': data.get('deletedBlobs', []),
            }
            (session_dir / 'session.json').write_text(json.dumps(meta, indent=2))

            # Learn from labeled blobs: update color map + histograms
            for blob in meta['blobs']:
                if blob.get('partKey') and blob.get('avgR') is not None:
                    color_id = blob['partKey'].split('_')[1]
                    update_learned_color(set_num, color_id, blob['avgR'], blob['avgG'], blob['avgB'],
                                         blob.get('histR'), blob.get('histG'), blob.get('histB'))

            # Learn optimal segmentation threshold from cropped photo + labeled blobs
            cropped_path = session_dir / 'cropped.jpg'
            labeled_blobs = [b for b in meta['blobs'] if b.get('partKey')]
            threshold_info = ''
            if cropped_path.exists() and labeled_blobs:
                t = compute_threshold(str(cropped_path), labeled_blobs)
                if t is not None:
                    update_learned_threshold(set_num, t)
                    lt = learned_threshold[set_num]
                    threshold_info = f', threshold={lt["threshold"]} ({lt["sampleCount"]} samples)'

            labeled = len(labeled_blobs)
            total = len(meta['blobs'])
            deleted = len(meta['deletedBlobs'])
            print(f'  session: {set_num} — {labeled}/{total} labeled, {deleted} deleted{threshold_info}')
            self.json_response({'ok': True, 'path': str(session_dir)})

        except Exception as e:
            print(f'  session save error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_reject(self):
        """Learn from rejected auto-assign: remove the wrong crop from index."""
        try:
            import base64
            data = json.loads(self.read_body())
            set_num = data['setNum']
            wrong_key = data['wrongKey']
            jpeg_b64 = data.get('jpeg')

            # Remove the most recent entry for this wrong key (likely the auto-assign crop)
            removed = 0
            with index_lock:
                entries = index.get(set_num, [])
                if entries:
                    # Remove the last entry matching the wrong key
                    for i in range(len(entries) - 1, -1, -1):
                        if entries[i]['key'] == wrong_key:
                            entries.pop(i)
                            removed += 1
                            break

            print(f'  reject: {set_num} {wrong_key} (removed {removed} from index)')
            self.json_response({'ok': True, 'removed': removed})

        except Exception as e:
            print(f'  reject error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_reset(self):
        """Reset all have-counts for a set in saved state."""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']

            if STATE_FILE.exists():
                state = json.loads(STATE_FILE.read_text())
                if 'sets' in state and set_num in state['sets']:
                    s = state['sets'][set_num]
                    for key in s.get('have', {}):
                        s['have'][key] = 0
                    STATE_FILE.write_text(json.dumps(state))
                    print(f'  reset: {set_num} — all counts zeroed')

            self.json_response({'ok': True})
        except Exception as e:
            print(f'  reset error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_debug(self):
        """Leave-one-out ML analysis for all crops in a set."""
        set_num = self.path.split('/api/debug/')[1]
        with index_lock:
            entries = index.get(set_num, [])
        if not entries:
            self.json_response({'crops': [], 'learnedColors': {}, 'stats': {}})
            return

        vecs = np.stack([e['vec'] for e in entries])
        n = len(entries)

        # Pairwise L2 distance matrix
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b  (vecs are normalized so ||v||=1)
        dots = vecs @ vecs.T
        norms_sq = np.sum(vecs ** 2, axis=1)
        dist_matrix = np.sqrt(np.maximum(norms_sq[:, None] + norms_sq[None, :] - 2 * dots, 0))
        np.fill_diagonal(dist_matrix, 999)

        crops = []
        for i in range(n):
            key = entries[i]['key']
            path = entries[i]['path']

            # Extract timestamp from filename
            try:
                ts = int(Path(path).stem.split('_')[-1])
            except:
                ts = 0

            # Leave-one-out: top suggestions excluding self
            dists = dist_matrix[i]
            sorted_idx = np.argsort(dists)
            suggestions = []
            seen = set()
            for j in sorted_idx:
                sk = entries[j]['key']
                if sk not in seen:
                    seen.add(sk)
                    suggestions.append({'key': sk, 'dist': round(float(dists[j]), 4)})
                if len(suggestions) >= 5:
                    break

            color_id = key.split('_')[1]
            guess_color = suggestions[0]['key'].split('_')[1] if suggestions else None

            # Make path relative to server root
            try:
                rel_path = str(Path(path).relative_to(Path(__file__).parent))
            except ValueError:
                rel_path = path

            is_catalog = entries[i].get('catalog', False)
            crops.append({
                'key': key,
                'colorId': color_id,
                'path': rel_path,
                'timestamp': ts,
                'suggestions': suggestions,
                'correct': suggestions[0]['key'] == key if suggestions else False,
                'guessColorId': guess_color,
                'catalog': is_catalog,
            })

        # Sort by timestamp descending (catalog entries have ts=0, go to end)
        crops.sort(key=lambda x: -x['timestamp'])

        colors = learned_colors.get(set_num, {})
        threshold = learned_threshold.get(set_num, {}).get('threshold', 200)
        unique_keys = sorted(set(e['key'] for e in entries))
        user_crops = [c for c in crops if not c.get('catalog')]
        stats = {
            'total': len(user_crops),
            'correct': sum(1 for c in user_crops if c['correct']),
            'uniqueParts': len(unique_keys),
            'catalogCount': sum(1 for c in crops if c.get('catalog')),
        }
        self.json_response({
            'crops': crops,
            'learnedColors': colors,
            'stats': stats,
            'threshold': threshold,
            'keys': unique_keys,
        })

    def handle_dedup(self):
        """Remove duplicate crops from index and disk. Keep newest per unique vector."""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']

            with index_lock:
                entries = index.get(set_num, [])
                if not entries:
                    self.json_response({'ok': True, 'removed': 0})
                    return

                vecs = np.stack([e['vec'] for e in entries])
                n = len(entries)

                # Group by vector hash (round to avoid float noise)
                seen = {}
                to_remove = []
                # Process newest first (later timestamps = later in list usually)
                for i in range(n - 1, -1, -1):
                    vec_hash = tuple(np.round(vecs[i], 6))
                    if vec_hash in seen:
                        to_remove.append(i)
                    else:
                        seen[vec_hash] = i

                # Remove duplicates from index and optionally from disk
                removed_count = 0
                for i in sorted(to_remove, reverse=True):
                    path = entries[i]['path']
                    entries.pop(i)
                    # Delete file
                    try:
                        Path(path).unlink(missing_ok=True)
                    except:
                        pass
                    removed_count += 1

                index[set_num] = entries

            print(f'  dedup: {set_num} — removed {removed_count}, kept {len(entries)}')
            self.json_response({'ok': True, 'removed': removed_count, 'remaining': len(entries)})
        except Exception as e:
            print(f'  dedup error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_delete_crop(self):
        """Delete a single crop from index and disk."""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']
            crop_path = data['path']  # relative path

            abs_path = str(Path(__file__).parent / crop_path)

            removed = False
            with index_lock:
                entries = index.get(set_num, [])
                for i, e in enumerate(entries):
                    if e['path'] == abs_path:
                        entries.pop(i)
                        removed = True
                        break

            if removed:
                try:
                    Path(abs_path).unlink(missing_ok=True)
                except:
                    pass

            print(f'  delete-crop: {crop_path} (removed={removed})')
            self.json_response({'ok': True, 'removed': removed})

        except Exception as e:
            print(f'  delete-crop error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_relabel(self):
        """Relabel a crop: change its part key in the index and rename the file."""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']
            old_path = data['path']  # relative path like data/crops/42078-1/...
            new_key = data['newKey']  # e.g. "3707_0"

            # Resolve to absolute path
            abs_path = str(Path(__file__).parent / old_path)

            with index_lock:
                entries = index.get(set_num, [])
                found = None
                for e in entries:
                    if e['path'] == abs_path:
                        found = e
                        break

                if not found:
                    self.json_response({'error': 'Crop not found in index'}, 404)
                    return

                old_key = found['key']
                found['key'] = new_key

                # Rename file: {partNum}_{colorId}_{timestamp}.jpg
                old_file = Path(abs_path)
                if old_file.exists():
                    # Extract timestamp from old filename
                    ts = old_file.stem.split('_')[-1]
                    new_parts = new_key.split('_')
                    new_filename = f'{new_parts[0]}_{new_parts[1]}_{ts}.jpg'
                    new_file = old_file.parent / new_filename
                    old_file.rename(new_file)
                    found['path'] = str(new_file)

            print(f'  relabel: {old_key} → {new_key} ({old_path})')
            self.json_response({'ok': True, 'oldKey': old_key, 'newKey': new_key})

        except Exception as e:
            print(f'  relabel error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_bootstrap(self):
        """Download catalog images and add to index for a set.
        Accepts { setNum, parts: [{key, imgUrl}, ...] }.
        Skips parts already in catalog dir."""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']
            parts = data['parts']

            cat_dir = CATALOG_DIR / set_num
            cat_dir.mkdir(parents=True, exist_ok=True)

            # Check which catalog images we already have
            existing = set(f.stem for f in cat_dir.glob('*.jpg'))

            to_download = [p for p in parts if p['key'] not in existing and p.get('imgUrl')]
            if not to_download:
                with index_lock:
                    cat_count = sum(1 for e in index.get(set_num, []) if e.get('catalog'))
                self.json_response({'ok': True, 'downloaded': 0, 'total': cat_count})
                return

            # Respond quickly — do downloads in background
            self.json_response({'ok': True, 'downloading': len(to_download)})

            def download_catalog():
                downloaded = 0
                failed = 0
                for p in to_download:
                    key = p['key']
                    url = p['imgUrl']
                    filepath = cat_dir / f'{key}.jpg'
                    try:
                        req = urllib.request.Request(url, headers={'User-Agent': 'LostBricks/1.0'})
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            img_bytes = resp.read()
                        filepath.write_bytes(img_bytes)
                        vec = img_to_vec(img_bytes)
                        np.save(str(filepath) + '.vec.npy', vec)
                        add_to_index(set_num, key, vec, str(filepath))
                        downloaded += 1
                    except Exception as e:
                        failed += 1
                        if failed <= 3:
                            print(f'  catalog skip {key}: {e}')
                print(f'  bootstrap {set_num}: {downloaded} downloaded, {failed} failed')

            threading.Thread(target=download_catalog, daemon=True).start()

        except Exception as e:
            print(f'  bootstrap error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_flag(self):
        """Flag a crop for review."""
        try:
            data = json.loads(self.read_body())
            flags = []
            if FLAGS_FILE.exists():
                flags = json.loads(FLAGS_FILE.read_text())
            flags.append({
                'path': data.get('path'),
                'key': data.get('key'),
                'note': data.get('note', ''),
                'timestamp': int(time.time() * 1000),
            })
            FLAGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            FLAGS_FILE.write_text(json.dumps(flags, indent=2))
            print(f'  flag: {data.get("key")} — {data.get("note", "")}')
            self.json_response({'ok': True, 'flagCount': len(flags)})
        except Exception as e:
            self.json_response({'error': str(e)}, 400)


# ── Main ──

if __name__ == '__main__':
    load_cnn()
    load_learned()

    print('Loading crop index (CNN embeddings)...')
    t0 = time.time()
    load_index()
    total = sum(len(v) for v in index.values())
    print(f'Index ready: {total} crops across {len(index)} sets ({time.time()-t0:.1f}s)')

    # Bootstrap histograms if missing
    needs_hist = any(
        'histR' not in c
        for colors in learned_colors.values()
        for c in colors.values()
        if c.get('count', 0) > 0
    )
    if needs_hist:
        print('Rebuilding histograms from crop images...')
        rebuild_histograms()

    # Build color classifier training data + train
    print('Training color classifier...')
    build_color_training_data()
    train_color_classifier()

    httpd = ReuseHTTPServer(('0.0.0.0', PORT), Handler)

    # HTTPS if certs exist and --no-tls not set
    cert = Path(__file__).parent / 'cert.pem'
    key = Path(__file__).parent / 'key.pem'
    if not NO_TLS and cert.exists() and key.exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert), str(key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f'Serving HTTPS on port {PORT}')
    else:
        print(f'Serving HTTP on port {PORT} (no certs found)')

    httpd.serve_forever()
