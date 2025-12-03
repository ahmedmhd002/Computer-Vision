
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

