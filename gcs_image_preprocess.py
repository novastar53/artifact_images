import io
import os
import sys
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any

import ray
from google.cloud import storage
from PIL import Image, ImageOps
from tqdm import tqdm


# ---------------------------------------
# Utilities
# ---------------------------------------

def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    # gcs://bucket/prefix -> ("bucket", "prefix")
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected URI like gs://bucket[/prefix], got: {uri}")
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def ensure_suffix(path: str) -> str:
    """Ensure the given path ends with a slash."""
    return path if path.endswith("/") else path + "/"


def safe_im_open(b: bytes) -> Optional[Image.Image]:
    try:
        im = Image.open(io.BytesIO(b))
        im.load()  # force load to catch early errors
        return im
    except Exception:
        return None


def to_format_extension(fmt: str) -> str:
    # Pillow formats to common file extensions
    fmt = (fmt or "JPEG").upper()
    if fmt == "JPEG":
        return "jpg"
    if fmt == "PNG":
        return "png"
    if fmt == "WEBP":
        return "webp"
    return fmt.lower()


# ---------------------------------------
# Config
# ---------------------------------------

@dataclass
class PipelineConfig:
    input_uri: str
    output_uri: str
    resize_w: int
    resize_h: int
    crop_w: int
    crop_h: int
    thumb_size: int
    output_format: str = "JPEG"  # "JPEG", "PNG", "WEBP"
    jpeg_quality: int = 90
    overwrite: bool = False


# ---------------------------------------
# Ray remote helpers
# ---------------------------------------

@ray.remote(num_cpus=0)
class GCSClientActor:
    """A tiny actor to reuse a single storage.Client across tasks on each worker."""
    def __init__(self):
        self.client = storage.Client()

    def read_blob(self, bucket: str, name: str) -> Optional[bytes]:
        try:
            b = self.client.bucket(bucket).blob(name)
            return b.download_as_bytes()
        except Exception as e:
            import traceback
            print(f"[GCS-READ-ERROR] gs://{bucket}/{name}: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def write_blob(self, bucket: str, name: str, data: bytes, content_type: Optional[str] = None) -> bool:
        try:
            blob = self.client.bucket(bucket).blob(name)
            if content_type:
                blob.content_type = content_type
            # Add an explicit timeout; upload_from_string infers bytes vs str
            blob.upload_from_string(data, content_type=content_type, timeout=120)
            return True
        except Exception as e:
            import traceback
            print(f"[GCS-WRITE-ERROR] gs://{bucket}/{name}: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def blob_exists(self, bucket: str, name: str) -> bool:
        try:
            return self.client.bucket(bucket).blob(name).exists()
        except Exception as e:
            import traceback
            print(f"[GCS-EXISTS-ERROR] gs://{bucket}/{name}: {e}", file=sys.stderr)
            traceback.print_exc()
            raise


@ray.remote
def process_one(
    conf: PipelineConfig,
    gcs: ray.actor.ActorHandle,
    in_bucket: str,
    in_key: str,
    out_bucket: str,
    out_prefix: str,
) -> Dict[str, Any]:
    """Download one image, apply transforms, upload three variants, return a small report dict."""
    try:
        raw = ray.get(gcs.read_blob.remote(in_bucket, in_key))
    except Exception as e:
        print(f"[ERROR] {in_key} -> download_error: {e}")
        return {"key": in_key, "status": "download_error"}
    if raw is None:
        print(f"[ERROR] {in_key} -> download_error (empty)")
        return {"key": in_key, "status": "download_error"}

    im = safe_im_open(raw)
    if im is None:
        print(f"[ERROR] {in_key} -> not_an_image")
        return {"key": in_key, "status": "not_an_image"}

    # Convert to RGB to avoid mode issues (e.g., "P", "RGBA" to RGB for JPEG)
    if conf.output_format.upper() == "JPEG":
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")

    # 1) Resize (preserve aspect ratio, fit within box)
    resized = ImageOps.contain(im, (conf.resize_w, conf.resize_h), method=Image.Resampling.LANCZOS)

    # 2) Center-crop to target
    crop_w, crop_h = conf.crop_w, conf.crop_h
    # If the resized image is smaller than crop, pad first using ImageOps.fit to cover edge cases.
    cropped = ImageOps.fit(resized, (crop_w, crop_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    # 3) Square thumbnail (keep center crop, maintain aspect)
    thumb = ImageOps.fit(im, (conf.thumb_size, conf.thumb_size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    # Determine output file base
    base_name = os.path.splitext(os.path.basename(in_key))[0]
    ext = to_format_extension(conf.output_format)
    out_paths = {
        "resized": f"{out_prefix}resized/{base_name}.{ext}",
        "cropped": f"{out_prefix}cropped/{base_name}.{ext}",
        "thumb":   f"{out_prefix}thumbnails/{base_name}.{ext}",
    }

    # Skip if already exists (unless overwrite)
    if not conf.overwrite:
        exists = [ray.get(gcs.blob_exists.remote(out_bucket, p)) for p in out_paths.values()]
        if all(exists):
            return {"key": in_key, "status": "skipped_exists", "outputs": out_paths}
    
    print(out_paths)

    # Serialize and upload
    def _save_and_upload(image: Image.Image, dest_path: str) -> bool:
        print(f"Uploading image to {out_bucket, dest_path}")
        buf = io.BytesIO()
        params = {}
        if conf.output_format.upper() == "JPEG":
            params = dict(quality=conf.jpeg_quality, optimize=True)
        image.save(buf, format=conf.output_format, **params)
        data = buf.getvalue()
        # Set a reasonable content-type
        ct = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp"
        }.get(conf.output_format.upper(), "application/octet-stream")
        try:
            return ray.get(gcs.write_blob.remote(out_bucket, dest_path, data, content_type=ct))
        except Exception as e:
            print(f"[UPLOAD-ERROR] {in_key} -> gs://{out_bucket}/{dest_path}: {e}", file=sys.stderr)
            return False

    ok1 = _save_and_upload(resized, out_paths["resized"])
    ok2 = _save_and_upload(cropped, out_paths["cropped"])
    ok3 = _save_and_upload(thumb,   out_paths["thumb"])

    print("Upload results:", ok1, ok2, ok3)

    status = "ok" if (ok1 and ok2 and ok3) else "partial_fail"

    if status == "ok":
        print(f"[SUCCESS] {in_key} processed")
    else:
        print(f"[ERROR] {in_key} -> {status}")

    return {"key": in_key, "status": status, "outputs": out_paths if status == "ok" else None}


# ---------------------------------------
# Listing input images
# ---------------------------------------

def list_input_keys(input_uri: str) -> Tuple[str, list]:
    client = storage.Client()
    in_bucket, in_prefix = parse_gcs_uri(input_uri)
    bucket = client.bucket(in_bucket)
    # list_blobs is efficient and paginated internally
    keys = []
    try:
        iterator = bucket.list_blobs(prefix=in_prefix)
    except Exception as e:
        print(f"[GCS-LIST-ERROR] gs://{in_bucket}/{in_prefix}: {e}", file=sys.stderr)
        raise
    for blob in iterator:
        # You can filter by extension if desired:
        if blob.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")):
            keys.append(blob.name)
    return in_bucket, keys


# ---------------------------------------
# Main
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ray + GCS image preprocessing pipeline.")
    parser.add_argument("--input", required=True, help="Input GCS URI, e.g., gs://my-bucket/images/")
    parser.add_argument("--output", required=True, help="Output GCS URI, e.g., gs://processed-bucket/processed/")
    parser.add_argument("--resize", default="1024x768", help="Resize WxH, e.g., 1024x768")
    parser.add_argument("--crop", default="1024x768", help="Center crop WxH, e.g., 1024x768")
    parser.add_argument("--thumb", default="256", help="Square thumbnail side length (pixels), e.g., 256")
    parser.add_argument("--format", default="JPEG", choices=["JPEG", "PNG", "WEBP"], help="Output image format")
    parser.add_argument("--jpeg_quality", type=int, default=90, help="JPEG quality (only for JPEG)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs if they already exist")
    parser.add_argument("--num_workers", type=int, default=0, help="Ray num_cpus override (0 = auto)")
    args = parser.parse_args()

    resize_w, resize_h = map(int, args.resize.lower().split("x"))
    crop_w, crop_h = map(int, args.crop.lower().split("x"))
    thumb_size = int(args.thumb)

    conf = PipelineConfig(
        input_uri=args.input,
        output_uri=args.output,
        resize_w=resize_w,
        resize_h=resize_h,
        crop_w=crop_w,
        crop_h=crop_h,
        thumb_size=thumb_size,
        output_format=args.format,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )

    runtime_env = {}
    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac:
        runtime_env["env_vars"] = {"GOOGLE_APPLICATION_CREDENTIALS": gac}

    # Ray init
    if args.num_workers and args.num_workers > 0:
        ray.init(num_cpus=args.num_workers, runtime_env=runtime_env or None)
    else:
        ray.init(runtime_env=runtime_env or None)

    out_bucket, out_prefix = parse_gcs_uri(conf.output_uri)
    #out_prefix = ensure_suffix(out_prefix)

    in_bucket, keys = list_input_keys(conf.input_uri)
    if not keys:
        print("No matching image files found in input prefix.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Discovered {len(keys)} images. Spinning up workers...")

    gcs = GCSClientActor.options(name="gcs_client", lifetime="detached").remote()

    # Submit tasks in parallel
    futures = [
        process_one.remote(conf, gcs, in_bucket, key, out_bucket, out_prefix)
        for key in keys
    ]

    # Collect results with a progress bar
    results = []
    for r in tqdm(ray.get(futures), total=len(futures), desc="Processing"):
        results.append(r)

    # Basic summary
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped_exists")
    failed = len(results) - ok - skipped

    print("\nSummary")
    print("-------")
    print(f"OK: {ok}")
    print(f"Skipped (exists): {skipped}")
    print(f"Failed/Partial: {failed}")

    # Optional: write a small CSV report to stdout
    try:
        import csv
        writer = csv.DictWriter(sys.stdout, fieldnames=["key", "status"])
        print("\nCSV Report:\n")
        writer.writeheader()
        for r in results:
            writer.writerow({"key": r["key"], "status": r["status"]})
    except Exception:
        pass


if __name__ == "__main__":
    main()