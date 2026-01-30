import argparse
from pathlib import Path

from PIL import Image


def convert_tiffs(input_dir: Path, output_dir: Path, recursive: bool) -> int:
    if recursive:
        tiff_files = list(input_dir.rglob("*.tif")) + list(input_dir.rglob("*.tiff"))
    else:
        tiff_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))

    if not tiff_files:
        print(f"No .tif/.tiff files found in {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0

    for tiff_path in tiff_files:
        rel_path = tiff_path.relative_to(input_dir)
        target_dir = output_dir / rel_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        png_path = target_dir / (tiff_path.stem + ".png")

        with Image.open(tiff_path) as img:
            img.save(png_path, format="PNG")
        converted += 1

    print(f"Converted {converted} file(s) to {output_dir}")
    return converted


def main() -> None:
    default_input = Path(
        r"C:\Users\gdf01\Documents\Segmentation\data\images\newimages_3\masks"
    )

    parser = argparse.ArgumentParser(description="Convert .tif/.tiff masks to .png.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing .tif/.tiff files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_input,
        help="Directory to write .png files.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Convert .tif/.tiff files recursively.",
    )

    args = parser.parse_args()
    convert_tiffs(args.input_dir, args.output_dir, args.recursive)


if __name__ == "__main__":
    main()
