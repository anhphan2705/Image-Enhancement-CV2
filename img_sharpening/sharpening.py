import cv2
import numpy as np

# Read img file
img = cv2.imread("./img_data/cat.jpg")

# Create custom kernel
kernel1 = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
kernel2 = np.array([[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]])

# img sharpening using filter2D
img_filter2D = cv2.filter2D(img, ddepth=-1, kernel=kernel2)

# highpass filtering
img_gaussian_blur = cv2.GaussianBlur(img, (3, 3), sigmaX=3)
img_highpass = img - img_gaussian_blur + 127

# unsharp masking / Highboost filtering
img_highboost = cv2.addWeighted(img, 2, img_gaussian_blur, -1, 0)

# Laplacian filtering
img_laplacian_blur = cv2.Laplacian(img, ddepth=8)
img_laplacian_sharp = cv2.addWeighted(img, 1.3, img_laplacian_blur, -1, 0)

# print result
cv2.imshow("before", img)
cv2.waitKey()
cv2.imshow("after", img_laplacian_sharp)
cv2.waitKey()
cv2.destroyAllWindows()

# write result
# cv2.imwrite("./img_processed/img_filter2D.jpg", img_filter2D)
# cv2.imwrite("./img_processed/img_highboost.jpg", img_highboost)
# cv2.imwrite("./img_processed/img_laplacian_sharp.jpg", img_laplacian_sharp)





