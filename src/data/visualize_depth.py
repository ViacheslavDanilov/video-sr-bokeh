"""Visualize depth maps with proper normalization and colormap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


def visualize_depth_maps(
    input_dir: str = "data/depth/0000/tiff",
    output_dir: str = "data/depth/0000/viz",
    colormap: str = "turbo",
    save_individual: bool = True,
    create_grid: bool = True,
):
    """Visualize depth maps with proper normalization.

    Args:
        input_dir: Directory containing depth map TIFF files
        output_dir: Directory to save visualized images
        colormap: Matplotlib colormap to use (e.g., 'turbo', 'viridis', 'plasma', 'magma')
        save_individual: Save each depth map as individual PNG
        create_grid: Create a grid visualization of all depth maps
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all TIFF files
    tiff_files = sorted(input_path.glob("*.tif"))
    print(f"Found {len(tiff_files)} depth map files in {input_dir}")

    depth_maps = []
    for tiff_file in tiff_files:
        # Load depth map
        depth = tifffile.imread(tiff_file)

        # Convert to grayscale if multi-channel (take first channel or average)
        if len(depth.shape) == 3:
            # Use the first channel or average all channels
            depth = depth[:, :, 0]  # or use: depth.mean(axis=2)

        depth_maps.append((tiff_file.stem, depth))

        print(
            f"  {tiff_file.name}: shape={depth.shape}, min={depth.min():.2f}, max={depth.max():.2f}",
        )

    # Find global min/max for consistent normalization across all images
    all_depths = np.concatenate([d.flatten() for _, d in depth_maps])
    global_min, global_max = all_depths.min(), all_depths.max()
    print(f"\nGlobal range: [{global_min:.2f}, {global_max:.2f}]")

    # Save individual visualizations
    if save_individual:
        print(f"\nSaving individual visualizations to {output_dir}/")
        for name, depth in depth_maps:
            # Normalize to 0-1 range
            depth_norm = (depth - global_min) / (global_max - global_min)

            # Apply colormap
            cmap = plt.get_cmap(colormap)
            depth_colored = cmap(depth_norm)

            # Save as PNG (convert to uint8)
            depth_uint8 = (depth_colored[:, :, :3] * 255).astype(np.uint8)
            output_file = output_path / f"{name}.png"
            plt.imsave(str(output_file), depth_uint8)
            print(f"  Saved: {output_file.name}")

    # Create grid visualization
    if create_grid and len(depth_maps) > 0:
        n_images = len(depth_maps)
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for idx, (name, depth) in enumerate(depth_maps):
            row, col = idx // cols, idx % cols
            ax = axes[row, col]

            # Normalize
            depth_norm = (depth - global_min) / (global_max - global_min)

            im = ax.imshow(depth_norm, cmap=colormap, vmin=0, vmax=1)
            ax.set_title(f"{name}", fontsize=10)
            ax.axis("off")

        # Hide empty subplots
        for idx in range(n_images, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis("off")

        plt.suptitle(f"Depth Maps Visualization", fontsize=14)
        plt.tight_layout()

        grid_file = output_path / f"depth_grid_{colormap}.png"
        plt.savefig(str(grid_file), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved grid visualization: {grid_file}")


def main():
    """Run visualization for all depth map directories."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize depth maps")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/depth/0000/tiff",
        help="Input directory containing depth map TIFF files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for visualizations (default: input_viz)",
    )
    parser.add_argument(
        "--colormap",
        "-c",
        type=str,
        default="turbo",
        choices=["turbo", "viridis", "plasma", "magma", "inferno", "cividis", "gray"],
        help="Colormap to use for visualization",
    )
    parser.add_argument(
        "--all-dirs",
        action="store_true",
        help="Process all subdirectories in data/depth/",
    )

    args = parser.parse_args()

    if args.all_dirs:
        # Process all subdirectories
        base_dir = Path("data/depth")
        subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
        for subdir in subdirs:
            if subdir.name.startswith("."):
                continue
            input_subdir = subdir / "tiff"
            output_dir = str(subdir / "viz")
            print(f"\n{'=' * 60}")
            print(f"Processing: {subdir}")
            print(f"{'=' * 60}")
            visualize_depth_maps(
                input_dir=str(input_subdir),
                output_dir=output_dir,
                colormap=args.colormap,
            )
    else:
        output_dir = args.output or args.input.replace("/tiff", "/viz")
        visualize_depth_maps(
            input_dir=args.input,
            output_dir=output_dir,
            colormap=args.colormap,
        )


if __name__ == "__main__":
    main()
