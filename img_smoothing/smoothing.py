# Resources: 
#   - imread and imwrite https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html
import cv2
import numpy as np

# Load images to be processed
img = cv2.imread("./img_data/net.jpg", flags=-1)

# Creating kernel
kernel1 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])/9
kernel2 = np.ones((3, 3), np.float32)/9        # kernel1 and kernel2 are the same thing

""" Using 2D Convolution (custom kernel) """
# filter2D(sourceImage, ddepth, kernel)
img_filter2d = cv2.filter2D(src=img, ddepth=-1, kernel=kernel1)

""" Using built-in function """
# Averaging:        blur(image, shapeOfTheKernel)
img_avg_blur = cv2.blur(img, (3, 3))            # This is exactly the same as the filter2d using kernel 1 or 2

# Gaussian Blur:    GaussianBlur(image, shapeOfTheKernel, sigmaX)
#                   It is the same as using filter2D with kernel 
#                           [1, 2, 1]
#                      1/16 [2, 4, 2]
#                           [1, 2, 1]
img_gaussian = cv2.GaussianBlur(img, (3, 3), sigmaX=0)

# Median Blur:      medianBlur(image, kernel size)
img_median = cv2.medianBlur(img, ksize=3)

# Bilateral Blur:   bilateralFilter(image, diameter, sigmaColor, sigmaSpace)
img_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Show result
cv2.imshow("Before", img)
cv2.waitKey()
cv2.imshow("After", img_bilateral)
cv2.waitKey()
cv2.destroyAllWindows()

# Save processed image
# cv2.imwrite("./img_processed/img_filter2d.jpg", img_filter2d)
# cv2.imwrite("./img_processed/img_avg_blur.jpg", img_avg_blur)
# cv2.imwrite("./img_processed/img_gaussian.jpg", img_gaussian)
# cv2.imwrite("./img_processed/img_median.jpg", img_median)

