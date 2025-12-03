import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.makedirs('step1_preprocessed', exist_ok=True)
os.makedirs('tiles', exist_ok=True)

rar_path = 'im.rar'

if not os.path.exists('2x2') and os.path.exists(rar_path):
    print(f"Found {rar_path}. Setting up extraction tool...")


    os.system('apt-get install unrar > /dev/null')

    print("Extracting files...")

    result = os.system(f'unrar x -o+ {rar_path} > /dev/null')

    if result == 0:
        print(" Unrar complete.")
    else:
        print(" Error extracting. Please ensure 'im.rar' is uploaded correctly.")

elif os.path.exists('2x2'):
    print(" Files appear to be already extracted.")

elif not os.path.exists(rar_path):
    print(f" Warning: '{rar_path}' not found. Please upload it to Colab Files.")

print("\n--- CONFIGURATION ---")
print("Available configurations:")
print("1. 2x2 Grid")
print("2. 4x4 Grid")
print("3. 8x8 Grid")

choice = input("Select grid size (1, 2, or 3): ").strip()

if choice == '1':
    folder_name = '2x2'
    GLOBAL_GRID_SIZE = 2
elif choice == '2':
    folder_name = '4x4'
    GLOBAL_GRID_SIZE = 4
elif choice == '3':
    folder_name = '8x8'
    GLOBAL_GRID_SIZE = 8
else:
    print("Invalid selection. Defaulting to 2x2.")
    folder_name = '2x2'
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
