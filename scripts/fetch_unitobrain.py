#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 OpenPerfusion contributors
"""Pull a few patient folders out of the 80 GB UniToBrain .7z without downloading the whole thing.

Usage (Terminal, from any folder):
    python3 fetch_unitobrain.py --list                 # show what is in the archive and how it is packed
    python3 fetch_unitobrain.py --extract 2            # fetch the first 2 patient folders into ./unitobrain/
    python3 fetch_unitobrain.py --extract 2 --dry-run  # say how much it would download, fetch nothing

The IEEE DataPort download link is built in (DEFAULT_URL). It is a redirector that hands out a
time-limited Amazon S3 link; the script follows it once, uses the S3 link for the range requests, and
re-follows it if the S3 link expires mid-way. Pass a different URL as the first argument to override
(an https link, or s3://bucket/key which is tried as the public S3 address).

How it works: a .7z keeps its table of contents at the end of the file, so the script reads that with
HTTP range requests, then decompresses only the solid block that holds the requested folders, reading
the archive in 16 MB pieces as the decompressor asks for them. Nothing else is downloaded.
Needs: python3 and `pip3 install py7zr`.
"""
import argparse, io, os, ssl, sys, time, urllib.request, urllib.error
from collections import OrderedDict

try:
    import py7zr, py7zr.compressor
except ImportError:
    sys.exit("py7zr is missing: run  pip3 install py7zr  (or  python3 -m pip install --user py7zr)  and retry")

# py7zr feeds its decompressor 1 MB of input per file by default, which for thousands of small DICOM
# files means reading far past what is needed; 64 KB keeps the reads close to the compressed size.
try:
    py7zr.compressor.get_default_blocksize = lambda: 65536
except Exception:          # older py7zr: works, just downloads more than needed
    pass

DEFAULT_URL = ("https://ieee-dataport.org/dataport/s3-download-url/4470"
               "?key=b3Blbi82Njc5Ni9VbmlUT0JyYWluX0RlZXBIZWFsdGhfSUVFRS43eg%3D%3D&redirect=1")
CHUNK = int(os.environ.get("FETCH_CHUNK", 16 * 1024 * 1024))   # bytes per range request


def _install_https_opener():
    """python.org builds of Python on macOS ship without the system root certificates; use certifi's
    bundle when it is installed, otherwise the default store."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx)))


_install_https_opener()

CERT_HELP = ("TLS certificate check failed. Either run  pip3 install certifi  and retry, or double-click "
             "'Install Certificates.command' in the Python folder under /Applications (python.org builds of "
             "Python need that once).")


class RangeFile(io.RawIOBase):
    """A read-only, seekable file over an HTTP URL using Range requests, with a small chunk cache."""

    def __init__(self, url, chunk=CHUNK, cache_chunks=8):
        super().__init__()
        self.source, self.chunk, self.pos = url, chunk, 0
        self.cache = OrderedDict(); self.cache_chunks = cache_chunks
        self.bytes_fetched = 0; self.requests = 0
        self.url, self.size = self._resolve()

    def _resolve(self):
        """Follow the download link to the real file (an S3 link), check it supports ranges, get its size."""
        req = urllib.request.Request(self.source, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                final = r.geturl(); status = r.status
                ctype = r.headers.get("Content-Type", "")
                crange = r.headers.get("Content-Range", "")
                body = r.read(2048)
        except urllib.error.URLError as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                sys.exit(CERT_HELP)
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
                sys.exit(f"HTTP {e.code}: the link needs a login or has expired. Open the IEEE DataPort page in the "
                         f"browser you are logged in with, press Download, cancel the download, then copy the real "
                         f"address from the browser's downloads list (Safari: right-click the item, Copy Address; "
                         f"Chrome: chrome://downloads shows it) and run this script with that address in quotes.")
            raise
        if "text/html" in ctype or status != 206 or not crange:
            sys.exit("the link answered with a web page instead of the file (probably a login page). Open the IEEE "
                     "DataPort page in the browser you are logged in with, press Download, cancel it, and copy the "
                     "real address from the browser's downloads list; run the script with that address in quotes.")
        size = int(crange.rsplit("/", 1)[1])
        if final != self.source:
            print(f"download link resolved to {final.split('?')[0]} (time-limited; re-resolved automatically if it expires)")
        return final, size

    def _fetch(self, idx):
        if idx in self.cache:
            self.cache.move_to_end(idx); return self.cache[idx]
        a = idx * self.chunk; b = min(a + self.chunk, self.size) - 1
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise IOError("server ignored the Range header")
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code in (401, 403) and self.url != self.source:   # presigned S3 link expired: re-resolve once
                    print("S3 link expired; re-resolving the download link...")
                    self.url, _ = self._resolve(); continue
                if attempt == 5: raise
                time.sleep(2 * (attempt + 1))
            except Exception:                           # transient network error: retry
                if attempt == 5: raise
                time.sleep(2 * (attempt + 1))
        self.requests += 1; self.bytes_fetched += len(data)
        self.cache[idx] = data
        if len(self.cache) > self.cache_chunks: self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        out = []
        while n > 0 and self.pos < self.size:
            idx, off = divmod(self.pos, self.chunk)
            piece = self._fetch(idx)[off: off + n]
            if not piece: break
            out.append(piece); self.pos += len(piece); n -= len(piece)
        return b"".join(out)

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos

    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def writable(self): return False
    def close(self): pass


def human(n): return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default=DEFAULT_URL)
    ap.add_argument("--list", action="store_true", help="list top-level folders and solid blocks")
    ap.add_argument("--extract", type=int, default=0, metavar="N", help="extract the first N patient folders")
    ap.add_argument("--names", nargs="*", default=None, help="extract these top-level folder names instead")
    ap.add_argument("--out", default="unitobrain")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--grep", default=None, help="with --list: show every patient-level folder containing this text, and the distinct folder types")
    a = ap.parse_args()

    url = a.url
    if url.startswith("s3://"):
        bucket, key = url[5:].split("/", 1)
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        print(f"trying the public S3 address {url}")
    f = RangeFile(url)
    print(f"archive size {human(f.size)}; reading the table of contents from the end of the file...")
    z = py7zr.SevenZipFile(f, mode="r")
    entries = z.list()
    # "patient folders" = the shallowest directory level that has several siblings (the archive may
    # have a single root folder above them)
    tops, depth = [], 1
    for depth in range(1, 5):
        seen = []
        for e in entries:
            parts = e.filename.split("/")
            if len(parts) > depth:
                key = "/".join(parts[:depth])
                if key not in seen: seen.append(key)
        tops = seen
        if len(seen) >= 5: break
    folders = z.header.main_streams.unpackinfo.folders
    packsizes = z.header.main_streams.packinfo.packsizes
    print(f"{len(entries)} entries, {len(tops)} patient-level folders (depth {depth}), {len(folders)} solid block(s), "
          f"fetched {human(f.bytes_fetched)} so far")
    if a.list or (not a.extract and not a.names):
        import re
        kinds = {}
        for i, t in enumerate(tops):
            k = re.sub(r"[A-Z]+-\d+", "<id>", t.split("/")[-1])
            kinds.setdefault(k, [0, i]); kinds[k][0] += 1
        print("distinct folder types (count, first position in archive order):")
        for k, (n, i) in kinds.items(): print(f"    {n:5d}  #{i:<5d} {k}")
        if a.grep:
            print(f"folders containing '{a.grep}':")
            for i, t in enumerate(tops):
                if a.grep in t: print(f"    #{i:<5d} {t}")
        else:
            print("first patient-level folders in archive order:")
            for t in tops[:15]: print("   ", t)
        print("solid blocks (packed size):", ", ".join(human(p) for p in packsizes[:10]), "..." if len(packsizes) > 10 else "")
        print("A patient folder is extracted by decompressing its solid block from the block's start up to that folder,\n"
              "so the first folders in archive order are the cheapest to fetch.")
        return

    want = a.names if a.names else tops[:a.extract]
    targets = [e.filename for e in entries if "/".join(e.filename.split("/")[:depth]) in want and not e.is_directory]
    # cost estimate: which solid block holds each target, and how far into it
    # (py7zr maps files to folders in order; estimate = packed size of the blocks touched)
    file_folder = {}
    try:
        fi = 0
        ssi = getattr(z.header.main_streams, "substreamsinfo", None)
        counts = list(ssi.num_unpackstreams_folders) if ssi is not None else [1] * len(folders)
        for folder_idx, n in enumerate(counts):
            for _ in range(n):
                while fi < len(entries) and entries[fi].is_directory:   # directories take no stream
                    fi += 1
                if fi < len(entries):
                    file_folder[entries[fi].filename] = folder_idx; fi += 1
        blocks = sorted({file_folder.get(t, 0) for t in targets if t in file_folder})
        est = sum(packsizes[b] for b in blocks if b < len(packsizes))
    except Exception:
        blocks, est = ["?"], sum(packsizes)
    print(f"extracting {len(want)} folder(s): {', '.join(want)}")
    print(f"{len(targets)} files, in solid block(s) {blocks}; worst-case download about {human(est)}")
    if a.dry_run:
        return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    z.extract(path=a.out, targets=targets)
    z.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min; {f.requests} range requests, {human(f.bytes_fetched)} downloaded; "
          f"files are under {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
