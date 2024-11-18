cd data/external/era5/raw && (for f in *.zip; do unzip "$f" -d "${f%.zip}"; done)
