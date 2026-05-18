import shutil
from pathlib import Path
from PIL import Image
import argparse
import re

VALID_IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".tiff"}

def RenameSamples(dir, base_name="image", replace=False, output_dir=None, new_ext=None):
    """
    Rename images in a directory to [base_name]_[i].[ext]

    Args:
        dir (str or Path) : Directory containing the images.
        base_name (str) : Base name for renamed images.
        replace (bool) : If True, renames images in-place.
        output_dir (str or Path) [Optional] : Output directory i replace=False.
        new_ext (str) [Optional] : convert to a desired image extension such as
            jpg or png. Uses the original extension unless specified otherwise.

    Raises:
        ValueError : If no images are found or multiple extensions are detected.

    Note:
        For multiple extensions scenario, use new_ext parameter to set a uniform
        extension.
    """

    image_dir = Path(dir)
    assert image_dir.exists(), f"Directory does not exist: {image_dir}"
    assert image_dir.is_dir(), f"Not a directory: {image_dir}"

    # Determine output directory
    if replace: 
        target_dir = image_dir
    else: 
        if output_dir:
            target_dir = Path(output_dir)  
        else:
            raise ValueError("Output directory not provided when replace = False.")
        target_dir.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    existing_indices = []
    for f in target_dir.iterdir():
        match = pattern.match(f.stem)
        if match:
            existing_indices.append(int(match.group(1)))
    next_idx = max(existing_indices) + 1 if existing_indices else 0

    # Load files (basic image check)
    all_files = sorted(
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS
    )

    files_to_rename = [f for f in all_files if not pattern.match(f.stem)]
    if not files_to_rename:
        print("No new images to process.")
        return
    
    if new_ext:
        new_ext = new_ext.lower().lstrip(".")
        if new_ext == "jpg": new_ext = "jpeg"
        final_ext = new_ext
    else:
        # Get extensions of files we are actually going to process
        extensions = {f.suffix.lower() for f in files_to_rename}
        if len(extensions) > 1:
            print(f"Warning: Multiple extensions detected {extensions}. Defaulting to first encountered.")
        final_ext = files_to_rename[0].suffix.lower().lstrip(".")

    # Process files
    for i, src in enumerate(files_to_rename):
        current_num = next_idx + i
        dst = target_dir / f"{base_name}_{current_num}.{final_ext}"
        
        if new_ext:
            with Image.open(src) as img:
                img = img.convert("RGB")
                img.save(dst, format=final_ext.upper())
            if replace:
                src.unlink()
        else:
            if replace:
                # Use rename logic; if src and dst are same, it's a no-op
                if src != dst:
                    src.rename(dst)
            else:
                shutil.copy2(src, dst)
    
    print(f"Renamed {len(files_to_rename)} images starting from index {next_idx}.")
    print(f"Final format: .{final_ext}")
    print(f"Renamed files saved to {target_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenameSamples")
    parser.add_argument("dir", type=str, help="Directory containing images")
    parser.add_argument("--base_name", type=str, default="image", help="Base filename (default: image)")
    parser.add_argument("--replace", action="store_true", help="Rename images in-place (destructive)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory if not replace")
    parser.add_argument("--new_ext", type=str, default=None, help="Convert images to this format")
    args = parser.parse_args()

    assert not (args.replace and args.output_dir), "Cannot use --replace and --output_dir together"

    RenameSamples(
        dir=args.dir,
        base_name=args.base_name,
        replace=args.replace,
        output_dir=args.output_dir,
        new_ext=args.new_ext
    )