# =========================================================
#  Stochastic Finance & Valuation Platform — Makefile
# =========================================================

.PHONY: help install run test lint clean

# Default target
help:
	@echo ""
	@echo "  Stochastic Finance & Valuation Platform"
	@echo "  ========================================="
	@echo "  make install   Install all Python dependencies"
	@echo "  make run       Launch the Streamlit dashboard"
	@echo "  make test      Run the full pytest test suite"
	@echo "  make lint      Check code style with flake8"
	@echo "  make clean     Remove __pycache__ and .pytest_cache"
	@echo ""

# Install dependencies into the active virtual environment
install:
	pip install -r requirements.txt

# Launch the Streamlit app
run:
	streamlit run app.py

# Run all unit tests with verbose output
test:
	python -m pytest -v

# Lint source and test files
lint:
	flake8 src/ tests/ ui/ app.py --max-line-length=120 --ignore=E501,W503

# Remove Python cache artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned up cache files."
