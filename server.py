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
from pathlib import Path

import numpy as np
from PIL import Image

# ── Config ──

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
NO_TLS = '--no-tls' in sys.argv
CROP_DIR = Path(__file__).parent / 'data' / 'crops'
STATE_FILE = Path(__file__).parent / 'data' / 'state.json'
SESSION_DIR = Path(__file__).parent / 'data' / 'sessions'
FEATURE_SIZE = 32  # resize crops to 32x32 → 3072-dim vector

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


def img_to_vec(img_bytes):
    """JPEG bytes → normalized 3072-dim numpy vector."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((FEATURE_SIZE, FEATURE_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr


def load_index():
    """Scan data/crops on startup, build in-memory index."""
    if not CROP_DIR.exists():
        return
    for set_dir in CROP_DIR.iterdir():
        if not set_dir.is_dir():
            continue
        set_num = set_dir.name
        entries = []
        for f in set_dir.glob('*.jpg'):
            # filename: {partNum}_{colorId}_{timestamp}.jpg
            parts = f.stem.split('_')
            if len(parts) < 3:
                continue
            key = f'{parts[0]}_{parts[1]}'
            try:
                vec = img_to_vec(f.read_bytes())
                entries.append({'key': key, 'vec': vec, 'path': str(f)})
            except Exception as e:
                print(f'  skip {f}: {e}')
        if entries:
            index[set_num] = entries
            print(f'  loaded {len(entries)} crops for set {set_num}')


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


def update_learned_color(set_num, color_id, avg_r, avg_g, avg_b):
    """Update running average of photographed color for a LEGO color."""
    if set_num not in learned_colors:
        learned_colors[set_num] = {}
    colors = learned_colors[set_num]
    cid = str(color_id)
    if cid not in colors:
        colors[cid] = {'r': avg_r, 'g': avg_g, 'b': avg_b, 'count': 1}
    else:
        c = colors[cid]
        n = c['count']
        # Running average
        c['r'] = round((c['r'] * n + avg_r) / (n + 1), 1)
        c['g'] = round((c['g'] * n + avg_g) / (n + 1), 1)
        c['b'] = round((c['b'] * n + avg_b) / (n + 1), 1)
        c['count'] = n + 1
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


def add_to_index(set_num, key, vec, path):
    """Add a new entry to the in-memory index."""
    with index_lock:
        if set_num not in index:
            index[set_num] = []
        index[set_num].append({'key': key, 'vec': vec, 'path': path})


def suggest(set_num, query_vec, top_k=10):
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

            # Add to index
            key = f'{part_num}_{color_id}'
            add_to_index(set_num, key, vec, str(filepath))

            # Learn color from blob's average RGB if provided
            avg_r = data.get('avgR')
            avg_g = data.get('avgG')
            avg_b = data.get('avgB')
            if avg_r is not None and avg_g is not None and avg_b is not None:
                update_learned_color(set_num, color_id, avg_r, avg_g, avg_b)

            count = len(index.get(set_num, []))
            colors = learned_colors.get(set_num, {})
            self.json_response({'ok': True, 'cropCount': count, 'learnedColors': colors})
            print(f'  label: {set_num} {key} ({count} total for set)')

        except Exception as e:
            print(f'  label error: {e}')
            self.json_response({'error': str(e)}, 400)

    def handle_suggest(self):
        """Suggest parts for a blob crop: { setNum, jpeg: base64 } → ranked keys."""
        try:
            data = json.loads(self.read_body())
            set_num = data['setNum']
            jpeg_b64 = data['jpeg']

            import base64
            jpeg_bytes = base64.b64decode(jpeg_b64)
            query_vec = img_to_vec(jpeg_bytes)

            results = suggest(set_num, query_vec)
            # Include learned colors for this set
            colors = learned_colors.get(set_num, {})
            with index_lock:
                crop_count = len(index.get(set_num, []))
            threshold = learned_threshold.get(set_num, {}).get('threshold')
            self.json_response({
                'suggestions': results,
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

            # Learn from labeled blobs: update color map
            for blob in meta['blobs']:
                if blob.get('partKey') and blob.get('avgR') is not None:
                    color_id = blob['partKey'].split('_')[1]
                    update_learned_color(set_num, color_id, blob['avgR'], blob['avgG'], blob['avgB'])

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


# ── Main ──

if __name__ == '__main__':
    print('Loading crop index...')
    load_index()
    total = sum(len(v) for v in index.values())
    print(f'Index ready: {total} crops across {len(index)} sets')
    load_learned()

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
