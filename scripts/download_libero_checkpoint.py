from openpi.shared import download

print(
    download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero/"), flush=True
)
