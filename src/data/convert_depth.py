#!/usr/bin/env python3
"""Convert TIFF depth maps to PNG format.

This script converts 32-bit TIFF depth maps to 8-bit grayscale PNG files.
The depth values are normalized across all images for consistent scaling.

Usage:
    # Convert a single sequence
    python src/data/convert_depth.py --input data/depth/0000/png

    # Convert all sequences
    python src/data/convert_depth.py --all-dirs

    # Invert depth (closer = brighter)
    python src/data/convert_depth.py --input data/depth/0000/png --invert
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def convert_tiff_to_png(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    invert: bool = False,
) -> None:
    """Convert TIFF depth maps to 8-bit PNG format.

    Args:
        input_dir: Directory containing TIFF depth maps.
        output_dir: Output directory for PNG files. If None, creates 'png' sibling folder.
        invert: If True, invert depth values (closer objects become brighter).
    """
    input_path = Path(input_dir)

    # Default output directory: sibling 'png' folder
    if output_dir is None:
        output_path = input_path.parent / "png"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    # Find all TIFF files
    tiff_files = sorted(input_path.glob("*.tif")) + sorted(input_path.glob("*.tiff"))

    if not tiff_files:
        print(f"No TIFF files found in {input_path}")
        return

    print(f"Found {len(tiff_files)} TIFF files in {input_path}")
    print(f"Output directory: {output_path}")

    # First pass: find global min/max for normalization
    global_min = float("inf")
    global_max = float("-inf")

    for tiff_file in tiff_files:
        depth = tifffile.imread(tiff_file)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        global_min = min(global_min, float(depth.min()))
        global_max = max(global_max, float(depth.max()))

    print(f"Depth range: [{global_min:.4f}, {global_max:.4f}]")

    # Second pass: convert each file
    for tiff_file in tiff_files:
        # Load depth map
        depth = tifffile.imread(tiff_file)

        # Handle multi-channel images
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # Normalize to 0-1 range
        if (global_max - global_min) > 0:
            depth_norm = (depth - global_min) / (global_max - global_min)
        else:
            depth_norm = np.zeros_like(depth)

        # Invert if requested (closer = brighter)
        if invert:
            depth_norm = 1.0 - depth_norm

        # Scale to 8-bit range
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        # Save as PNG
        output_file = output_path / f"{tiff_file.stem}.png"
        Image.fromarray(depth_uint8).save(output_file)

    print(f"Converted {len(tiff_files)} files to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Convert TIFF depth maps to 8-bit PNG format",
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/depth/0000/tiff"),
        help="Input directory containing TIFF depth maps",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for PNG files (default: sibling 'png' folder)",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert depth values (closer objects become brighter)",
    )
    parser.add_argument(
        "--all-dirs",
        action="store_true",
        help="Process all sequences in data/depth/",
    )

    args = parser.parse_args()

    if args.all_dirs:
        base_dir = Path("data/depth")
        if not base_dir.exists():
            print(f"Error: {base_dir} does not exist")
            return

        subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        for subdir in subdirs:
            if subdir.name.startswith("."):
                continue

            maps_dir = subdir / "tiff"
            if not maps_dir.exists():
                print(f"Skipping {subdir.name}: no 'tiff' folder")
                continue

            print(f"\n{'=' * 60}")
            print(f"Processing: {subdir.name}")
            print(f"{'=' * 60}")

            convert_tiff_to_png(
                input_dir=maps_dir,
                output_dir=subdir / "png",
                invert=args.invert,
            )
    else:
        convert_tiff_to_png(
            input_dir=args.input,
            output_dir=args.output,
            invert=args.invert,
        )


if __name__ == "__main__":
    main()
