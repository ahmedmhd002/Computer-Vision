import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import zipfile
import os

os.makedirs('step1_preprocessed', exist_ok=True)
os.makedirs('tiles', exist_ok=True)

# --- ZIP EXTRACTION LOGIC ---
def extract_if_needed(zip_filename, target_folder):
    if not os.path.exists(target_folder):
        if os.path.exists(zip_filename):
            print(f"Extracting {zip_filename}...")
            try:
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    zip_ref.extractall('.')
                print(f" Extracted {zip_filename} successfully.")
            except zipfile.BadZipFile:
                print(f" Error: {zip_filename} is not a valid zip file.")
        else:
            print(f" Note: {zip_filename} not found. '{target_folder}' might be missing.")
    else:
        print(f" Folder '{target_folder}' already exists. Skipping extraction.")

# Check and extract all puzzle zips
extract_if_needed('puzzle_2x2.zip', 'puzzle_2x2')
extract_if_needed('puzzle_4x4.zip', 'puzzle_4x4')
extract_if_needed('puzzle_8x8.zip', 'puzzle_8x8')

# Fallback for im.rar (legacy support)
# Only try if NO puzzle folders exist
if not os.path.exists('puzzle_2x2') and not os.path.exists('2x2') and os.path.exists('im.rar'):
    print("\nAttempting legacy extraction from 'im.rar'...")
    if os.name != 'nt':
        os.system('apt-get install unrar > /dev/null')
    result = os.system(f'unrar x -o+ im.rar > /dev/null')
    if result != 0:
        print(" Manual extraction required for 'im.rar'.")


print("\n--- CONFIGURATION ---")
print("Available configurations:")
print("1. 2x2 Grid")
print("2. 4x4 Grid")
print("3. 8x8 Grid")

choice = input("Select grid size (1, 2, or 3): ").strip()

if choice == '1':
    folder_name = 'puzzle_2x2' if os.path.exists('puzzle_2x2') else '2x2'
    GLOBAL_GRID_SIZE = 2
elif choice == '2':
    folder_name = 'puzzle_4x4' if os.path.exists('puzzle_4x4') else '4x4'
    GLOBAL_GRID_SIZE = 4
elif choice == '3':
    folder_name = 'puzzle_8x8' if os.path.exists('puzzle_8x8') else '8x8'
    GLOBAL_GRID_SIZE = 8
else:
    print(f"Invalid selection: '{choice}'. Defaulting to 2x2.")
    folder_name = 'puzzle_2x2' if os.path.exists('puzzle_2x2') else '2x2'
    GLOBAL_GRID_SIZE = 2

print(f"\nSelected Grid: {GLOBAL_GRID_SIZE}x{GLOBAL_GRID_SIZE} (Folder: {folder_name})")

image_id = input("Enter the image number to process (e.g., 0): ").strip()
image_path = os.path.join(folder_name, f"{image_id}.jpg")

print(f"Looking for: {image_path}")
original_image = cv2.imread(image_path)

if original_image is None:
    print(f" Error: Could not load image. Check if '{folder_name}/{image_id}.jpg' exists.")
else:
    print(f" Success! Loaded Image {image_id} from {folder_name}.")
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(original_rgb)
    plt.title(f"Input: {folder_name}/{image_id}.jpg")
    plt.axis('off')
    plt.show()






    #################################################################



    
target_width = 800
aspect_ratio = original_image.shape[0] / original_image.shape[1]
target_height = int(target_width * aspect_ratio)
img_resized = cv2.resize(original_image, (target_width, target_height))

gray_image = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)


gaussian_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

cv2.imwrite('step1_preprocessed/preprocessed_gray.jpg', gaussian_blurred)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Grayscale")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gaussian_blurred, cmap='gray')
plt.title("Step 1 Output: Gaussian Blur")
plt.axis('off')
plt.show()

print(" Preprocessing complete. Image saved.")











############################################################






clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(gaussian_blurred)


bilateral = cv2.bilateralFilter(enhanced_image, 9, 75, 75)

kernel_sharpening = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
sharpened = cv2.filter2D(bilateral, -1, kernel_sharpening)


plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(enhanced_image, cmap='gray')
plt.title("CLAHE")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(bilateral, cmap='gray')
plt.title("Bilateral Filter")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(sharpened, cmap='gray')
plt.title("Step 2 Output: Sharpened")
plt.axis('off')
plt.show()

print(" Enhancement complete.")




############################################################



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

print(f" Slicing complete. {len(tiles)} tiles saved.")





###################################################################






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

print(" Improved Contour Extraction complete.")












##################################################################




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



#####################################








