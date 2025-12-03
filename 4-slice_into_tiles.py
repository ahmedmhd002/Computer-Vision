
rows = GLOBAL_GRID_SIZE
cols = GLOBAL_GRID_SIZE

height, width = sharpened.shape
tile_height = height // rows
tile_width = width // cols

tiles = []
tile_coords = []

print(f"Slicing image into {rows}x{cols} grid (Total {rows*cols} tiles)...")

for r in range(rows):
    for c in range(cols):
        y_start = r * tile_height
        y_end = (r + 1) * tile_height
        x_start = c * tile_width
        x_end = (c + 1) * tile_width

        tile_img = sharpened[y_start:y_end, x_start:x_end]
        tiles.append(tile_img)
        tile_coords.append((r, c))

        filename = f'tiles/tile_{r}_{c}.jpg'
        cv2.imwrite(filename, tile_img)


plt.figure(figsize=(12, 12)) 

print(f"Displaying all {len(tiles)} tiles...")

for i, tile in enumerate(tiles):

    plt.subplot(rows, cols, i + 1)
    plt.imshow(tile, cmap='gray')

    if rows < 8:
        plt.title(f"{tile_coords[i]}")

    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Member 3: Slicing complete. {len(tiles)} tiles saved.")
