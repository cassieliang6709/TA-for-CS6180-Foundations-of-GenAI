# HPC Hugging Face Training Guide

This repository contains a comprehensive guide and utility scripts for training Hugging Face models on High Performance Computing (HPC) clusters.

## Contents

- **[End-to-End Guide](index.html)**: A complete HTML guide covering environment setup, cache configuration, and training steps tailored for SLURM-based HPC systems (specifically configured for Northeastern University's Discovery cluster).
- **Scripts**:
  - `scripts/train_hf_hpc.py`: A ready-to-run Python script for fine-tuning GPT-2, including robust cache handling.
  - `scripts/test_hf_inference.py`: A quick verification script to test GPU availability and Hugging Face model loading.

## Usage

1. **View the Guide**: Open `index.html` in your web browser.
2. **Run Scripts**:
   ```bash
   # Verify setup
   python scripts/test_hf_inference.py
   
   # Run training
   python scripts/train_hf_hpc.py
   ```

## License

MIT
