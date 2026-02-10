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
CROP_DIR = Path(__file__).parent / 'data' / 'crops'
FEATURE_SIZE = 32  # resize crops to 32x32 → 3072-dim vector

# ── In-memory index ──
# { setNum: [ { key: "partNum_colorId", vec: np.array, path: str }, ... ] }
index = {}
index_lock = threading.Lock()


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

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/label':
            self.handle_label()
        elif self.path == '/api/suggest':
            self.handle_suggest()
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

            count = len(index.get(set_num, []))
            self.json_response({'ok': True, 'count': count})
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
            self.json_response({'suggestions': results})

        except Exception as e:
            print(f'  suggest error: {e}')
            self.json_response({'error': str(e)}, 400)


# ── Main ──

if __name__ == '__main__':
    print('Loading crop index...')
    load_index()
    total = sum(len(v) for v in index.values())
    print(f'Index ready: {total} crops across {len(index)} sets')

    httpd = http.server.HTTPServer(('0.0.0.0', PORT), Handler)

    # HTTPS if certs exist
    cert = Path(__file__).parent / 'cert.pem'
    key = Path(__file__).parent / 'key.pem'
    if cert.exists() and key.exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert), str(key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f'Serving HTTPS on port {PORT}')
    else:
        print(f'Serving HTTP on port {PORT} (no certs found)')

    httpd.serve_forever()
