# TEM nanoparticle analysis

# Run analysis on all images in images/ directory
run *ARGS:
    uv run python analise.py {{ ARGS }}

# Run on a specific image
analyze IMAGE:
    uv run python analise.py {{ IMAGE }}

# Run on all images in a directory with known scale
batch DIR SCALE_NM="200":
    uv run python analise.py {{ DIR }} --scale-nm {{ SCALE_NM }}

# Format code
fmt:
    uv run ruff format analise.py

# Lint code
lint:
    uv run ruff check analise.py

# Lint and fix
fix:
    uv run ruff check --fix analise.py
    uv run ruff format analise.py

# Clean output directory
clean:
    rm -rf output/
