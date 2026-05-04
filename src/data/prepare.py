"""Automated ShapeNet data preparation for CHOIR.

Downloads ShapeNet point clouds from AtlasNetV2, converts PLY to H5, and
generates the stability test NPZ. Run once before training:

    uv run python src/data/prepare.py [--data_dir data/shapenet]

Pipeline:
  1. Download ShapeNet PLY files from AtlasNetV2 (cloud.enpc.fr)
  2. Convert PLY → H5 (one per synset, centered point clouds)
  3. Generate stability evaluation NPZ (fixed rotations for reproducibility)
"""

import argparse
import os
import struct
import sys
import tarfile
import zipfile
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

DOWNLOAD_URL = "https://cloud.enpc.fr/s/JNf3NAxGbQoQsKY/download"

SYNSETS = [
    ("02691156", "plane"),
    ("02828884", "bench"),
    ("02933112", "cabinet"),
    ("02958343", "car"),
    ("03001627", "chair"),
    ("03211117", "monitor"),
    ("03636649", "lamp"),
    ("03691459", "speaker"),
    ("04090263", "firearm"),
    ("04256520", "couch"),
    ("04379243", "table"),
    ("04401088", "cellphone"),
    ("04530566", "watercraft"),
]

STABILITY_CLASSES = [0, 3, 4, 10]  # plane, car, chair, table


def _read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices and normals from a binary/ASCII PLY file.

    Returns:
        vertices: (N, 3) float32
        normals: (N, 3) float32 (zeros if not present)
    """
    with open(path, "rb") as f:
        # Parse header
        line = f.readline().decode().strip()
        assert line == "ply", f"Not a PLY file: {path}"

        fmt = None
        num_vertices = 0
        properties = []
        in_vertex = False

        while True:
            line = f.readline().decode().strip()
            if line == "end_header":
                break
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element" and parts[1] == "vertex":
                num_vertices = int(parts[2])
                in_vertex = True
            elif parts[0] == "element":
                in_vertex = False
            elif parts[0] == "property" and in_vertex:
                properties.append(parts[2])  # property name

        has_normals = all(n in properties for n in ("nx", "ny", "nz"))
        prop_indices = {name: i for i, name in enumerate(properties)}

        # Read vertex data
        if fmt == "binary_little_endian":
            dtype = np.dtype([(p, "<f4") for p in properties])
            data = np.frombuffer(f.read(num_vertices * dtype.itemsize), dtype=dtype)
        elif fmt == "binary_big_endian":
            dtype = np.dtype([(p, ">f4") for p in properties])
            data = np.frombuffer(f.read(num_vertices * dtype.itemsize), dtype=dtype)
        else:  # ascii
            rows = []
            for _ in range(num_vertices):
                vals = f.readline().decode().strip().split()
                rows.append([float(v) for v in vals])
            data_arr = np.array(rows, dtype=np.float32)
            # Convert to structured for uniform access
            data = np.zeros(num_vertices, dtype=[(p, "f4") for p in properties])
            for i, p in enumerate(properties):
                data[p] = data_arr[:, i]

        vertices = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float32)
        if has_normals:
            normals = np.column_stack([data["nx"], data["ny"], data["nz"]]).astype(np.float32)
        else:
            normals = np.zeros_like(vertices)

    return vertices, normals


def download_shapenet(data_dir: str) -> str:
    """Download ShapeNet point clouds from AtlasNetV2.

    Returns:
        Path to extracted customShapeNet directory.
    """
    import urllib.request
    import time

    raw_dir = os.path.join(data_dir, "_raw")
    os.makedirs(raw_dir, exist_ok=True)

    zip_path = os.path.join(raw_dir, "customShapeNet.zip")
    zip_part = zip_path + ".part"
    extract_dir = os.path.join(raw_dir, "customShapeNet")

    if os.path.isdir(extract_dir) and any(
        os.path.isdir(os.path.join(extract_dir, s)) for s, _ in SYNSETS
    ):
        print(f"Already downloaded: {extract_dir}")
        return extract_dir

    if not os.path.exists(zip_path):
        print(f"Downloading ShapeNet from {DOWNLOAD_URL} ...")
        print("(~12GB, supports resume on failure)")

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                # Resume from partial download
                existing_size = os.path.getsize(zip_part) if os.path.exists(zip_part) else 0

                req = urllib.request.Request(DOWNLOAD_URL)
                if existing_size > 0:
                    req.add_header("Range", f"bytes={existing_size}-")
                    print(f"  Resuming from {existing_size / (1024**2):.0f} MB (attempt {attempt}/{max_retries})")

                resp = urllib.request.urlopen(req, timeout=60)

                # Check if server supports range requests
                content_range = resp.headers.get("Content-Range")
                if existing_size > 0 and content_range is None:
                    # Server doesn't support resume, restart
                    existing_size = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing_size > 0 else "wb"

                total_size = int(resp.headers.get("Content-Length", 0)) + existing_size
                downloaded = existing_size

                with open(zip_part, mode) as f:
                    while True:
                        chunk = resp.read(1024 * 1024)  # 1MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = min(100, downloaded * 100 // total_size)
                            mb = downloaded / (1024**2)
                            total_mb = total_size / (1024**2)
                            sys.stdout.write(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)")
                            sys.stdout.flush()

                print()

                # Verify download completeness
                final_size = os.path.getsize(zip_part)
                if total_size > 0 and final_size < total_size:
                    raise IOError(f"Incomplete download: {final_size}/{total_size} bytes")

                # Rename to final path
                os.rename(zip_part, zip_path)
                print(f"  Download complete: {zip_path}")
                break

            except (urllib.error.URLError, IOError, TimeoutError, ConnectionError) as e:
                print(f"\n  Attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    wait = min(30, 5 * attempt)
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"Failed to download after {max_retries} attempts. "
                        f"Partial file saved at {zip_part}. "
                        f"Re-run to resume, or download manually from {DOWNLOAD_URL}"
                    )

    # Extract
    print(f"Extracting to {raw_dir} ...")
    if zipfile.is_zipfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
    elif tarfile.is_tarfile(zip_path):
        with tarfile.open(zip_path, "r:*") as tf:
            tf.extractall(raw_dir)
    else:
        raise RuntimeError(f"Unknown archive format: {zip_path}")

    # Find the customShapeNet directory (may be nested)
    for root, dirs, files in os.walk(raw_dir):
        for d in dirs:
            candidate = os.path.join(root, d)
            if any(os.path.isdir(os.path.join(candidate, s)) for s, _ in SYNSETS):
                if candidate != extract_dir:
                    os.rename(candidate, extract_dir)
                return extract_dir

    # If synset dirs are directly under a top-level dir
    for entry in os.listdir(raw_dir):
        candidate = os.path.join(raw_dir, entry)
        if os.path.isdir(candidate) and candidate != extract_dir:
            if any(os.path.isdir(os.path.join(candidate, s)) for s, _ in SYNSETS):
                os.rename(candidate, extract_dir)
                return extract_dir

    raise RuntimeError(f"Could not find ShapeNet synset directories in {raw_dir}")


def convert_ply_to_h5(ply_dir: str, data_dir: str) -> None:
    """Convert PLY files to H5 format (one file per synset)."""
    print("\nConverting PLY → H5 ...")

    for synset_id, name in SYNSETS:
        h5_path = os.path.join(data_dir, f"{synset_id}.h5")
        if os.path.exists(h5_path):
            print(f"  [{name}] {h5_path} already exists, skipping")
            continue

        synset_dir = os.path.join(ply_dir, synset_id)
        if not os.path.isdir(synset_dir):
            print(f"  [{name}] directory not found, skipping")
            continue

        # Find PLY files (may be in ply/ subdir or directly)
        ply_subdir = os.path.join(synset_dir, "ply")
        search_dir = ply_subdir if os.path.isdir(ply_subdir) else synset_dir
        ply_files = sorted([f for f in os.listdir(search_dir) if f.endswith(".ply")])

        if not ply_files:
            print(f"  [{name}] no PLY files found, skipping")
            continue

        print(f"  [{name}] {len(ply_files)} shapes → {h5_path}")

        with h5py.File(h5_path, "w") as f:
            pcd_grp = f.create_group("pcd")
            point_grp = pcd_grp.create_group("point")
            normal_grp = pcd_grp.create_group("normal")

            for j, ply_name in enumerate(tqdm(ply_files, desc=f"    {name}", leave=False)):
                ply_path = os.path.join(search_dir, ply_name)
                vertices, normals = _read_ply(ply_path)

                # Center (no scaling)
                vertices = vertices - vertices.mean(axis=0, keepdims=True)

                point_grp[str(j)] = vertices
                normal_grp[str(j)] = normals


def generate_stability_npz(
    data_dir: str,
    num_points: int = 10000,
    num_rotations: int = 10,
    seed: int = 0,
) -> None:
    """Generate stability evaluation NPZ with fixed random rotations."""
    npz_path = os.path.join(
        data_dir, f"preprocessed_stability_c=0-3-4-10_n={num_points}.npz"
    )
    if os.path.exists(npz_path):
        print(f"\nStability NPZ already exists: {npz_path}")
        return

    print(f"\nGenerating stability NPZ (seed={seed}) ...")
    np.random.seed(seed)

    # Load test split for stability classes
    synset_path = os.path.join(data_dir, "synsetoffset2category.txt")
    with open(synset_path) as fp:
        synset_dict = {
            i: line.rstrip("\n").split()[-1] for i, line in enumerate(fp.readlines())
        }

    ret = {"rotated_pcd": [], "rots": [], "label": []}

    for label_idx in STABILITY_CLASSES:
        synset_id = synset_dict[label_idx]
        h5_path = os.path.join(data_dir, f"{synset_id}.h5")
        assert os.path.exists(h5_path), f"H5 not found: {h5_path}"

        with h5py.File(h5_path, "r") as h5:
            keys = sorted(h5["pcd"]["point"].keys(), key=int)
            # Test split: last 20%
            split = int(0.8 * len(keys))
            test_keys = keys[split:]

            name = [n for s, n in SYNSETS if s == synset_id][0]
            print(f"  [{name}] {len(test_keys)} test shapes")

            for key in tqdm(test_keys, desc=f"    {name}", leave=False):
                pcd = np.asarray(h5["pcd"]["point"][key])
                pcd = pcd[:num_points]

                rots = Rotation.random(num_rotations).as_matrix().astype(np.float32)
                rot_pcds = np.stack(
                    [np.einsum("nj, ij -> ni", pcd, rot) for rot in rots]
                )

                ret["rotated_pcd"].append(rot_pcds)
                ret["rots"].append(rots)
                ret["label"].append(label_idx)

    ret["rotated_pcd"] = np.stack(ret["rotated_pcd"])
    ret["rots"] = np.stack(ret["rots"])
    ret["label"] = np.array(ret["label"])

    np.savez(npz_path, **ret)
    print(f"  Saved: {npz_path}")
    print(f"  Shapes: {ret['rotated_pcd'].shape}")


def main():
    parser = argparse.ArgumentParser(description="Prepare ShapeNet data for CHOIR")
    parser.add_argument(
        "--data_dir", type=str, default="data/shapenet",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--num_points", type=int, default=10000,
        help="Number of points for stability test",
    )
    parser.add_argument(
        "--num_rotations", type=int, default=10,
        help="Number of rotations per shape for stability test",
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip download (use existing PLY files)",
    )
    parser.add_argument(
        "--ply_dir", type=str, default=None,
        help="Path to existing customShapeNet directory (skips download)",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Step 1: Download
    if args.ply_dir:
        ply_dir = args.ply_dir
    elif args.skip_download:
        ply_dir = os.path.join(args.data_dir, "_raw", "customShapeNet")
    else:
        ply_dir = download_shapenet(args.data_dir)

    # Step 2: PLY → H5
    convert_ply_to_h5(ply_dir, args.data_dir)

    # Step 3: Stability NPZ
    generate_stability_npz(
        args.data_dir,
        num_points=args.num_points,
        num_rotations=args.num_rotations,
    )

    print("\nDone! Data is ready at:", args.data_dir)


if __name__ == "__main__":
    main()
