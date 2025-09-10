uv run gcs_image_preprocess.py \
  --input gs://artifact_images_happy_cat_234 \
  --output gs://artifact_thumbnails_happy_cat_234 \
  --resize 512x512 \
  --crop 128x128 \
  --thumb 256 \
  --format JPEG \
  --jpeg_quality 92