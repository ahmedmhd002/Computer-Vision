

final_dataset = []

print("Extracting features (Hu Moments)...")

for item in tile_contours_data:
    tile_id = item["id"]
    contours = item["contours"]

    tile_features = []

    for cnt in contours:
        moments = cv2.moments(cnt)

        if moments['m00'] == 0:
            continue

        hu_moments = cv2.HuMoments(moments).flatten()

        hu_moments_log = []
        for val in hu_moments:
            if val == 0:
                hu_moments_log.append(0)
            else:
                hu_moments_log.append(-1 * np.copysign(1.0, val) * np.log10(abs(val)))

        contour_list = cnt.tolist()

        tile_features.append({
            "contour_points": contour_list,
            "hu_moments": hu_moments_log
        })

    final_dataset.append({
        "tile_id": tile_id,
        "features": tile_features
    })

output_file = 'puzzle_features.json'
with open(output_file, 'w') as f:
    json.dump(final_dataset, f, indent=4)

print(f" Data packaged. Saved to '{output_file}'.")
print("\n--- JSON PREVIEW (First Contour of First Tile) ---")
print(json.dumps(final_dataset[0]['features'][0]['hu_moments'], indent=4))
