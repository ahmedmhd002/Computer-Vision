"""
Milestone 2: Puzzle Assembly
=============================
This script uses the preprocessed tiles from Milestone 1 to:
1. Extract edge profiles from each puzzle tile
2. Compare edges to find matching neighbors
3. Assemble the puzzle by placing tiles in the correct positions
4. Visualize the assembled result

No ML/AI - Pure classical computer vision using edge matching.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from itertools import permutations

# =============================================================================
# CONFIGURATION
# =============================================================================

print("\n" + "="*60)
print("MILESTONE 2: PUZZLE ASSEMBLY")
print("="*60)

# Ask user for grid configuration
print("\nAvailable configurations:")
print("1. 2x2 Grid")
print("2. 4x4 Grid")
print("3. 8x8 Grid")

choice = input("Select grid size (1, 2, or 3): ").strip()

if choice == '1':
    GRID_SIZE = 2
    folder_name = 'puzzle_2x2' if os.path.exists('puzzle_2x2') else '2x2'
elif choice == '2':
    GRID_SIZE = 4
    folder_name = 'puzzle_4x4' if os.path.exists('puzzle_4x4') else '4x4'
elif choice == '3':
    GRID_SIZE = 8
    folder_name = 'puzzle_8x8' if os.path.exists('puzzle_8x8') else '8x8'
else:
    print("Invalid selection. Defaulting to 2x2.")
    GRID_SIZE = 2
    folder_name = 'puzzle_2x2' if os.path.exists('puzzle_2x2') else '2x2'

NUM_TILES = GRID_SIZE * GRID_SIZE
print(f"\nGrid Size: {GRID_SIZE}x{GRID_SIZE} ({NUM_TILES} tiles)")

# =============================================================================
# STEP 1: LOAD TILES FROM MILESTONE 1
# =============================================================================

print("\n--- STEP 1: Loading Tiles ---")

tiles = []
tile_files = []

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        filename = f'tiles/tile_{r}_{c}.jpg'
        if os.path.exists(filename):
            tile = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            tiles.append(tile)
            tile_files.append(filename)
        else:
            print(f"Warning: {filename} not found!")

if len(tiles) != NUM_TILES:
    print(f"Error: Expected {NUM_TILES} tiles, found {len(tiles)}.")
    print("Please run Milestone 1 first to generate tiles.")
    exit(1)

print(f"Loaded {len(tiles)} tiles successfully.")

# =============================================================================
# STEP 2: EXTRACT EDGE PROFILES
# =============================================================================

print("\n--- STEP 2: Extracting Edge Profiles ---")

def extract_edge_profiles(tile, edge_width=5):
    """
    Extract edge profiles from a tile.
    Uses multiple rows/columns and averages them for robustness.
    
    Returns a dict with 'top', 'bottom', 'left', 'right' edge profiles.
    """
    # Average multiple rows/columns for noise reduction
    top = np.mean(tile[:edge_width, :], axis=0)
    bottom = np.mean(tile[-edge_width:, :], axis=0)
    left = np.mean(tile[:, :edge_width], axis=1)
    right = np.mean(tile[:, -edge_width:], axis=1)
    
    return {
        'top': top.astype(np.float32),
        'bottom': bottom.astype(np.float32),
        'left': left.astype(np.float32),
        'right': right.astype(np.float32)
    }

# Extract profiles for all tiles
tile_profiles = []
for i, tile in enumerate(tiles):
    profiles = extract_edge_profiles(tile)
    tile_profiles.append(profiles)
    
print(f"Extracted edge profiles for {len(tile_profiles)} tiles.")

# =============================================================================
# STEP 3: EDGE SIMILARITY COMPUTATION
# =============================================================================

print("\n--- STEP 3: Computing Edge Similarities ---")

def edge_similarity(edge1, edge2):
    """
    Compute similarity between two edges using normalized cross-correlation.
    Returns a value between -1 and 1 (higher = more similar).
    """
    # Normalize edges
    e1 = edge1 - np.mean(edge1)
    e2 = edge2 - np.mean(edge2)
    
    std1 = np.std(e1)
    std2 = np.std(e2)
    
    # Handle zero variance edges
    if std1 < 1e-8 or std2 < 1e-8:
        return 0.0
    
    e1 = e1 / std1
    e2 = e2 / std2
    
    # Normalized cross-correlation
    correlation = np.dot(e1, e2) / len(e1)
    return correlation

def compute_all_similarities():
    """
    Compute similarity matrices for all edge pairs.
    
    Returns:
    - right_left_sim[i][j]: similarity of tile i's RIGHT edge to tile j's LEFT edge
    - bottom_top_sim[i][j]: similarity of tile i's BOTTOM edge to tile j's TOP edge
    """
    n = len(tiles)
    
    # Right-to-Left matching (horizontal neighbors)
    right_left_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                right_left_sim[i][j] = edge_similarity(
                    tile_profiles[i]['right'],
                    tile_profiles[j]['left']
                )
    
    # Bottom-to-Top matching (vertical neighbors)
    bottom_top_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                bottom_top_sim[i][j] = edge_similarity(
                    tile_profiles[i]['bottom'],
                    tile_profiles[j]['top']
                )
    
    return right_left_sim, bottom_top_sim

right_left_sim, bottom_top_sim = compute_all_similarities()
print("Similarity matrices computed.")

# =============================================================================
# STEP 4: GREEDY PUZZLE ASSEMBLY
# =============================================================================

print("\n--- STEP 4: Assembling Puzzle ---")

def find_best_match(sim_matrix, source_idx, used_indices):
    """
    Find the best matching tile for a given source tile.
    Returns the index of the best match and the similarity score.
    """
    best_idx = -1
    best_score = -float('inf')
    
    for j in range(len(tiles)):
        if j not in used_indices and j != source_idx:
            score = sim_matrix[source_idx][j]
            if score > best_score:
                best_score = score
                best_idx = j
    
    return best_idx, best_score

def greedy_assembly():
    """
    Greedy assembly algorithm:
    1. Start with tile 0 in top-left corner
    2. For each row, find the best right neighbor
    3. For each new row, find the best tile below the first column
    """
    grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    used = set()
    confidence_scores = []
    
    # Try each tile as the starting tile and pick the one with best overall score
    best_grid = None
    best_total_score = -float('inf')
    
    for start_tile in range(len(tiles)):
        grid = [[-1 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        used = {start_tile}
        total_score = 0
        scores = []
        
        # Place starting tile
        grid[0][0] = start_tile
        
        # Fill the grid
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if r == 0 and c == 0:
                    continue  # Already placed
                
                if c > 0:
                    # Find tile that matches the left neighbor's right edge
                    left_neighbor = grid[r][c-1]
                    best_idx, score = find_best_match(right_left_sim, left_neighbor, used)
                    
                    if r > 0:
                        # Also consider matching with top neighbor
                        top_neighbor = grid[r-1][c]
                        _, top_score = -1, -float('inf')
                        
                        # Find candidate that matches both
                        candidates = []
                        for j in range(len(tiles)):
                            if j not in used:
                                h_score = right_left_sim[left_neighbor][j]
                                v_score = bottom_top_sim[top_neighbor][j]
                                combined = (h_score + v_score) / 2
                                candidates.append((j, combined, h_score, v_score))
                        
                        if candidates:
                            candidates.sort(key=lambda x: x[1], reverse=True)
                            best_idx = candidates[0][0]
                            score = candidates[0][1]
                else:
                    # First column: match with top neighbor
                    top_neighbor = grid[r-1][c]
                    best_idx, score = find_best_match(bottom_top_sim, top_neighbor, used)
                
                if best_idx != -1:
                    grid[r][c] = best_idx
                    used.add(best_idx)
                    total_score += score
                    scores.append(score)
        
        # Check if this is the best arrangement
        if total_score > best_total_score:
            best_total_score = total_score
            best_grid = [row[:] for row in grid]
            confidence_scores = scores[:]
    
    return best_grid, confidence_scores

assembled_grid, confidence_scores = greedy_assembly()

print("\nAssembled Grid (tile indices):")
for row in assembled_grid:
    print(row)

if confidence_scores:
    avg_confidence = np.mean(confidence_scores)
    print(f"\nAverage Match Confidence: {avg_confidence:.4f}")

# =============================================================================
# STEP 5: VISUALIZATION
# =============================================================================

print("\n--- STEP 5: Visualizing Results ---")

def visualize_assembly(grid, tiles, title="Assembled Puzzle"):
    """
    Visualize the assembled puzzle.
    """
    # Get tile dimensions
    tile_h, tile_w = tiles[0].shape
    
    # Create output image
    output_h = GRID_SIZE * tile_h
    output_w = GRID_SIZE * tile_w
    assembled_image = np.zeros((output_h, output_w), dtype=np.uint8)
    
    # Place tiles
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            tile_idx = grid[r][c]
            if tile_idx >= 0:
                y_start = r * tile_h
                y_end = (r + 1) * tile_h
                x_start = c * tile_w
                x_end = (c + 1) * tile_w
                assembled_image[y_start:y_end, x_start:x_end] = tiles[tile_idx]
    
    return assembled_image

# Create assembled image
assembled_image = visualize_assembly(assembled_grid, tiles)

# Show original tiles (scrambled order as loaded)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.title("Original Tiles (Input Order)")
scrambled = np.zeros_like(assembled_image)
tile_h, tile_w = tiles[0].shape
for i, tile in enumerate(tiles):
    r = i // GRID_SIZE
    c = i % GRID_SIZE
    scrambled[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tile
plt.imshow(scrambled, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Assembled Puzzle (After Matching)")
plt.imshow(assembled_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.savefig('milestone2_result.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n Result saved to 'milestone2_result.png'")

# =============================================================================
# STEP 6: DETAILED MATCHING REPORT
# =============================================================================

print("\n--- STEP 6: Matching Report ---")

print("\n=== NEIGHBOR RELATIONSHIPS ===")
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        tile_idx = assembled_grid[r][c]
        neighbors = []
        
        if c < GRID_SIZE - 1:
            right_idx = assembled_grid[r][c+1]
            score = right_left_sim[tile_idx][right_idx]
            neighbors.append(f"Right→{right_idx} ({score:.3f})")
        
        if r < GRID_SIZE - 1:
            bottom_idx = assembled_grid[r+1][c]
            score = bottom_top_sim[tile_idx][bottom_idx]
            neighbors.append(f"Down→{bottom_idx} ({score:.3f})")
        
        if neighbors:
            print(f"Tile {tile_idx} at ({r},{c}): {', '.join(neighbors)}")

# Save assembly data to JSON
assembly_data = {
    "grid_size": GRID_SIZE,
    "assembled_grid": assembled_grid,
    "average_confidence": float(np.mean(confidence_scores)) if confidence_scores else 0,
    "tile_positions": []
}

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        assembly_data["tile_positions"].append({
            "original_index": assembled_grid[r][c],
            "assembled_row": r,
            "assembled_col": c
        })

with open('assembly_result.json', 'w') as f:
    json.dump(assembly_data, f, indent=4)

print(f"\n Assembly data saved to 'assembly_result.json'")
print("\n" + "="*60)
print("MILESTONE 2 COMPLETE!")
print("="*60)
