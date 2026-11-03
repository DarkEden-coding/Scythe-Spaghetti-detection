"""Minimal EXR reader and PNG writer (stdlib + numpy), for use inside Blender.

Blender cannot hand a render's object-index pass back to Python without the
compositor, so `render_sequence` reads its own EXRs back: single-part
scanline files with NONE, ZIPS or ZIP compression, which is what Blender
writes for multilayer output.
"""

import struct
import zlib

import numpy as np

_TYPES = {0: np.dtype("<u4"), 1: np.dtype("<f2"), 2: np.dtype("<f4")}
_LINES = {0: 1, 2: 1, 3: 16}  # compression -> scanlines per chunk


def _attrs(data, pos):
    attrs = {}
    while data[pos] != 0:
        end = data.index(b"\0", pos)
        name = data[pos:end].decode()
        pos = end + 1
        end = data.index(b"\0", pos)
        kind = data[pos:end].decode()
        pos = end + 1
        size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        attrs[name] = (kind, data[pos:pos + size])
        pos += size
    return attrs, pos + 1


def _channels(raw):
    out, pos = [], 0
    while raw[pos] != 0:
        end = raw.index(b"\0", pos)
        name = raw[pos:end].decode()
        pixel_type = struct.unpack_from("<i", raw, end + 1)[0]
        out.append((name, _TYPES[pixel_type]))
        pos = end + 1 + 16
    return out


def _unzip(payload, size):
    raw = np.frombuffer(zlib.decompress(payload), dtype=np.uint8)
    if raw.size != size:
        raise ValueError("EXR chunk has the wrong size")
    # Undo the predictor, then the even/odd byte split.
    acc = np.cumsum(np.concatenate(([raw[0]], raw[1:].astype(np.int64) - 128)))
    t = (acc & 0xFF).astype(np.uint8)
    half = (raw.size + 1) // 2
    out = np.empty(raw.size, dtype=np.uint8)
    out[0::2] = t[:half]
    out[1::2] = t[half:]
    return out.tobytes()


def read_exr(path, want=None):
    """Return {channel name: 2D array} (top row first), from single-part or
    multipart scanline EXRs. `want` limits the channels decoded to those
    whose name contains any of the given strings."""
    with open(path, "rb") as f:
        data = f.read()
    if struct.unpack_from("<I", data, 0)[0] != 20000630:
        raise ValueError(f"{path} is not an EXR file")
    flags = struct.unpack_from("<I", data, 4)[0]
    if flags & 0x200:
        raise ValueError(f"{path}: tiled EXRs are not supported")
    multipart = bool(flags & 0x1000)
    pos = 8
    headers = []
    while True:
        attrs, pos = _attrs(data, pos)
        headers.append(attrs)
        if not multipart or data[pos] == 0:
            break
    if multipart:
        pos += 1
    parts = []
    for attrs in headers:
        channels = _channels(attrs["channels"][1])
        compression = attrs["compression"][1][0]
        if compression not in _LINES:
            raise ValueError(f"{path}: unsupported EXR compression {compression}")
        xmin, ymin, xmax, ymax = struct.unpack("<4i", attrs["dataWindow"][1])
        h = ymax - ymin + 1
        n_chunks = (struct.unpack("<i", attrs["chunkCount"][1])[0] if "chunkCount" in attrs
                    else (h + _LINES[compression] - 1) // _LINES[compression])
        offsets = struct.unpack_from(f"<{n_chunks}Q", data, pos)
        pos += 8 * n_chunks
        parts.append((channels, compression, (xmin, ymin, xmax, ymax), offsets))
    planes = {}
    for channels, compression, (xmin, ymin, xmax, ymax), offsets in parts:
        w, h = xmax - xmin + 1, ymax - ymin + 1
        keep = [want is None or any(s in name for s in want) for name, _ in channels]
        if not any(keep):
            continue
        out = {name: np.empty((h, w), dt) for (name, dt), k in zip(channels, keep) if k}
        row_bytes = sum(w * dt.itemsize for _, dt in channels)
        per = _LINES[compression]
        for off in offsets:
            if multipart:
                off += 4
            y, size = struct.unpack_from("<iI", data, off)
            payload = data[off + 8:off + 8 + size]
            rows = min(per, ymax - y + 1)
            expected = rows * row_bytes
            if size < expected:
                payload = _unzip(payload, expected)
            buf = np.frombuffer(payload, dtype=np.uint8)
            p = 0
            for r in range(rows):
                for (name, dt), k in zip(channels, keep):
                    nb = w * dt.itemsize
                    if k:
                        out[name][y - ymin + r] = buf[p:p + nb].view(dt)
                    p += nb
        planes.update(out)
    return planes


def find_channel(planes, *parts):
    """The one channel whose dot-separated name contains all `parts`."""
    hits = [n for n in planes if all(p in n.split(".") for p in parts)]
    if len(hits) != 1:
        raise KeyError(f"{parts}: {hits} in {sorted(planes)}")
    return planes[hits[0]]


def write_png(path, rgb):
    """Write an 8-bit RGB PNG from a (h, w, 3) uint8 array."""
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1)

    def chunk(tag, body):
        c = struct.pack(">I", len(body)) + tag + body
        return c + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
        f.write(chunk(b"IEND", b""))


def linear_to_srgb8(rgb, exposure=0.0):
    x = np.clip(np.nan_to_num(rgb.astype(np.float32)) * 2.0 ** exposure, 0, 1)
    x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (x * 255 + 0.5).astype(np.uint8)


# Distinct colours for class ids 0..8 in IndexOB previews.
CLASS_COLORS = np.array([
    (0, 0, 0), (70, 70, 90), (40, 110, 200), (60, 200, 60), (255, 40, 40),
    (255, 160, 0), (230, 60, 230), (0, 230, 230), (255, 255, 120)], np.uint8)
