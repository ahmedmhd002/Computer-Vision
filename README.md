# 🧩 Jigsaw Puzzle Solver

A Computer Vision project that automatically assembles scrambled puzzle tiles using classical image processing techniques.

## 📋 Project Overview

This project consists of two milestones:

| Milestone | Description | Script |
|-----------|-------------|--------|
| **Milestone 1** | Image preprocessing, enhancement, tile slicing & feature extraction | `simple_script.py` |
| **Milestone 2** | Edge matching & puzzle assembly | `milestone2_assembly.py` |

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy matplotlib
```

### Run Milestone 1 (Preprocessing)

```bash
python simple_script.py
```
- Select grid size (2x2, 4x4, or 8x8)
- Enter image number to process
- Outputs: preprocessed tiles in `tiles/` folder

### Run Milestone 2 (Assembly)

```bash
python milestone2_assembly.py
```
- Select same grid size as Milestone 1
- Automatically assembles tiles
- Outputs: `milestone2_result.png` and `assembly_result.json`

## 📁 Project Structure

```
project/
├── simple_script.py        # Milestone 1: Preprocessing pipeline
├── milestone2_assembly.py  # Milestone 2: Assembly pipeline
├── generate_report.py      # Report generator
├── puzzle_2x2/             # 2x2 puzzle images
├── puzzle_4x4/             # 4x4 puzzle images
├── puzzle_8x8/             # 8x8 puzzle images
├── tiles/                  # Extracted tiles (M1 output)
├── step1_preprocessed/     # Preprocessed images
├── puzzle_output/          # Visualization outputs
├── milestone2_result.png   # Assembly result
├── assembly_result.json    # Assembly data
└── puzzle_features.json    # Extracted features
```

## 🔧 Milestone 1: Preprocessing Pipeline

1. **Image Loading** - Load puzzle image from selected folder
2. **Preprocessing** - Resize, grayscale, Gaussian blur
3. **Enhancement** - CLAHE, bilateral filter, sharpening
4. **Tile Slicing** - Divide into N×N grid
5. **Contour Extraction** - Canny edge detection + morphological operations
6. **Feature Extraction** - Hu Moments computation

## 🧩 Milestone 2: Assembly Pipeline

1. **Load Tiles** - Import processed tiles from Milestone 1
2. **Edge Extraction** - Sample intensity profiles from tile edges
3. **Similarity Computation** - Normalized Cross-Correlation (NCC)
4. **Greedy Assembly** - Match tiles based on edge similarity
5. **Visualization** - Side-by-side comparison
6. **Report Generation** - JSON output with confidence scores

## 📊 Algorithm Details

### Edge Matching
- Extracts 1D intensity profiles from each edge (5-pixel depth averaging)
- Uses **Normalized Cross-Correlation (NCC)** for similarity scoring
- Range: [-1, 1] where 1 = perfect match

### Assembly Strategy
- **Greedy algorithm** with starting tile optimization
- Tries each tile as starting point
- Combines horizontal + vertical scores for interior tiles
- Achieves ~85% average matching confidence

## 📈 Results

| Metric | Value |
|--------|-------|
| Average Confidence | 84.79% |
| Supported Grids | 2×2, 4×4, 8×8 |
| Edge Profile Depth | 5 pixels |

## 👥 Team 32

| Name | ID |
|------|-----|
| Ahmed Mohamed Elmahdy Nagaty | 2100723 |
| Maria Ibraheem Nasseef Mikhaeil | 1901487 |
| Ahmed Elsayed Sayed Ahmed Mohmed | 20p7195 |
| Meena Maged Abdo Mekhaiel | 1900694 |
| Youssef Ashraf Mahmoud Othman Agha | 2301134 |



## 🛠️ Technologies

- **Python 3.x**
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **Matplotlib** - Visualization

## 📝 License

Computer Vision Course Project - December 2025
