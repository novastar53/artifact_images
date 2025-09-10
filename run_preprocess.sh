uv run gcs_image_preprocess.py \
  --input gs://artifact_images_happy_cat_234 \
  --output gs://artifact_thumbnails_happy_cat_234 \
  --resize 512x512 \
  --thumb 128 \
  --format JPEG \
  --jpeg_quality 92