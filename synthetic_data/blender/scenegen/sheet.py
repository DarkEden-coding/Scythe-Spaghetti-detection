"""Contact sheet for a rendered job: renders on top, IndexOB masks below,
one column per keyframe, for eyeballing a sequence at a glance."""

import numpy as np

from .imageio import write_png


def _read_png(path):
    """Read an 8-bit RGB PNG written by imageio.write_png."""
    import struct
    import zlib
    with open(path, "rb") as f:
        data = f.read()
    pos, idat, w, h = 8, b"", 0, 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 3 + 1)
    return raw[:, 1:].reshape(h, w, 3)


def contact_sheet(job_dir, spec, meta, out_path, width=400):
    tiles = []
    for k in meta["keyframes"]:
        img = _read_png(f"{job_dir}/k{k['idx']:03d}.png")
        mask = _read_png(f"{job_dir}/k{k['idx']:03d}_mask.png")
        step = max(1, img.shape[1] // width)
        img, mask = img[::step, ::step], mask[::step, ::step]
        sev = float(k["failure_state"].get("severity", 0.0))
        bar = np.zeros((6, img.shape[1], 3), np.uint8)
        bar[:, :int(sev * img.shape[1])] = (255, 60, 60)
        if k["events_active"]:
            bar[3:, :] = (255, 255, 0)
        tiles.append(np.concatenate([img, bar, mask], axis=0))
    gap = np.full((tiles[0].shape[0], 4, 3), 255, np.uint8)
    row = [tiles[0]]
    for t in tiles[1:]:
        row += [gap, t]
    write_png(out_path, np.concatenate(row, axis=1))
