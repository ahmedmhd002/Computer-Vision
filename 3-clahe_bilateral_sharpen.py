
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

