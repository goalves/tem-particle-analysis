# TEM nanoparticle analysis

# Run analysis on all images in images/ directory
run *ARGS:
    uv run python measure.py {{ ARGS }}

# Run on a specific image
analyze IMAGE:
    uv run python measure.py {{ IMAGE }}

# Run on all images in a directory (scale auto-detected per image)
batch DIR:
    uv run python measure.py {{ DIR }}

# Interactively annotate one image: click each particle, SAM segments it
annotate IMAGE:
    uv run --group annotate python annotate.py "{{ IMAGE }}"

# Format code
fmt:
    uv run ruff format measure.py

# Lint code
lint:
    uv run ruff check measure.py

# Lint and fix
fix:
    uv run ruff check --fix measure.py
    uv run ruff format measure.py

# Clean output directory
clean:
    rm -rf output/
