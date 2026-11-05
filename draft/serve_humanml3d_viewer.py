#!/usr/bin/env python3
"""Serve humanml3d_viewer.html with the HumanML3D clip index and motions on demand.

Example:
    python serve_humanml3d_viewer.py --port 8000

The server binds to localhost and only reads the dataset. Forward the port to open it remotely.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

import numpy as np

DRAFT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = DRAFT_DIR.parents[1]/'HumanML3D'/'HumanML3D'
VIEWER = DRAFT_DIR/'humanml3d_viewer.html'
SPLITS = ['train', 'val', 'test']
MOTION_PATH = re.compile(r'/api/motion/(M?\d{6})')
BODY_JOINTS = 22
R_HIP, L_HIP, R_SHOULDER, L_SHOULDER = 2, 1, 17, 16  # _humanml_features.face_joint_indx


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', type=Path, default=DEFAULT_ROOT,
                   help='Directory holding new_joints/, texts/ and the split lists.')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    return p.parse_args()


def read_splits(root):
    return {name: split for split in SPLITS for name in (root/(split+'.txt')).read_text().split()}


def read_captions(path):
    """Return captions, prefixed with their time range when they cover only part of the clip."""
    captions = []
    for line in path.read_text().splitlines():
        fields = line.split('#')
        start, end = np.nan_to_num([float(fields[2]), float(fields[3])])
        prefix = '[{:.1f}-{:.1f}s] '.format(start, end) if start or end else ''
        captions.append(prefix + fields[0].strip())
    return captions


def read_sources(repo_root):
    """Map a clip name to its source file, from index.csv."""
    with (repo_root/'index.csv').open(newline='') as f:
        return {Path(row['new_name']).stem: Path(row['source_path']) for row in csv.DictReader(f)}


def clip_entry(root, sources, splits, path):
    name = path.stem
    source = sources[name.lstrip('M')]
    return {
        'name': name,
        'subset': source.parts[1],  # pose_data/<subset>/...
        'split': splits.get(name, 'none'),
        'frames': int(np.load(path, mmap_mode='r').shape[0]),
        'captions': read_captions(root/'texts'/(name+'.txt')),
        'source': str(source),
    }


def build_index(root):
    sources = read_sources(root.parent)
    splits = read_splits(root)
    paths = sorted((root/'new_joints').glob('*.npy'))
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(lambda path: clip_entry(root, sources, splits, path), paths))


def face_forward(joints):
    """Turn a clip about the vertical axis so its first pose faces +z; heights are unchanged."""
    first = joints[0]
    across = (first[R_HIP]-first[L_HIP]) + (first[R_SHOULDER]-first[L_SHOULDER])
    forward_x, forward_z = across[2], -across[0]  # (0, 1, 0) x across
    angle = np.arctan2(forward_x, forward_z)
    cos, sin = np.cos(angle), np.sin(angle)
    turned = joints.copy()
    turned[..., 0] = joints[..., 0]*cos - joints[..., 2]*sin
    turned[..., 2] = joints[..., 0]*sin + joints[..., 2]*cos
    return turned


def read_motion(root, name, raw_height):
    """Return (T, 22, 3) joints: new_joints/, or joints/ before the per-clip floor shift."""
    if raw_height:
        path = root.parent/'joints'/(name+'.npy')
        return face_forward(np.load(path)[:, :BODY_JOINTS]) if path.is_file() else None
    path = root/'new_joints'/(name+'.npy')
    return np.load(path) if path.is_file() else None


def make_handler(root, index_gzip):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            path = url.path
            motion = MOTION_PATH.fullmatch(path)
            try:
                if path == '/':
                    self.reply(VIEWER.read_bytes(), 'text/html; charset=utf-8')
                elif path == '/api/index':
                    self.reply(index_gzip, 'application/json', {'Content-Encoding': 'gzip'})
                elif motion:
                    self.reply_motion(motion.group(1), parse_qs(url.query).get('height') == ['raw'])
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The browser cancels requests when the result page changes.

        def reply_motion(self, name, raw_height):
            joints = read_motion(root, name, raw_height)
            if joints is None:
                self.send_error(404, 'Unknown clip: '+name)
                return
            self.reply(joints.astype('<f4').tobytes(), 'application/octet-stream')

        def reply(self, body, content_type, headers=None):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def log_request(self, code='-', size='-'):
            pass  # Each result page requests 12 motions; errors are still logged.

    return Handler


def main():
    args = arguments()
    root = args.data_root.resolve()
    print('Indexing {} ...'.format(root), flush=True)
    index = build_index(root)
    index_gzip = gzip.compress(json.dumps(index, separators=(',', ':')).encode(), compresslevel=6)
    print('{} clips, index {:.1f} MB'.format(len(index), len(index_gzip)/1e6), flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(root, index_gzip))
    print('Open http://{}:{}/'.format(args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
