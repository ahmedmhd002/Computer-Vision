

tile_contours_data = []

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)

plt.figure(figsize=(12, 12))

for i, tile in enumerate(tiles):

    edges = auto_canny(tile)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)


    contours, hierarchy = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = []
    tile_area = tile.shape[0] * tile.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < 50:
            continue

        if area > 0.95 * tile_area:
            continue

        valid_contours.append(cnt)

    tile_contours_data.append({
        "id": i,
        "original_tile": tile,
        "contours": valid_contours
    })

    vis_tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis_tile, valid_contours, -1, (0, 255, 0), 2)

    plt.subplot(rows, cols, i + 1)
    plt.imshow(vis_tile)
    plt.title(f"Tile {i} Contours: {len(valid_contours)}")
    plt.axis('off')

plt.show()

print("Member 4: Improved Contour Extraction complete.")
