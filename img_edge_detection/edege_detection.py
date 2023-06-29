import cv2
import numpy as np

# img read
img = cv2.imread("./img_data/peppers.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (3,3), 0)                        # remove noise

# kernel
kernel_0deg = np.array([[-1 , -1, -1],
                        [0, 0, 0],
                        [1, 1, 1]])
kernel_90deg = np.array([[-1, 0, 1],
                        [-1, 0, 1],
                        [-1, 0, 1]])

# filter2D applying Prewitt operation
img_0deg = cv2.filter2D(img_gray, ddepth=-1, kernel=kernel_0deg)
img_90deg = cv2.filter2D(img_gray, ddepth=-1, kernel=kernel_90deg)
img_filter2D = cv2.addWeighted(img_0deg, 0.5, img_90deg, 0.5, 0) 

# Canny edge detection
img_canny = cv2.Canny(img, 100, 200)

# Sobel    VERY SENSITIVE TO IMAGE NOISE
img_sobelx = cv2.Sobel(img_gray, -1, 1, 0, ksize=3)
img_sobely = cv2.Sobel(img_gray, -1, 0, 1, ksize=3)                  
img_sobel = cv2.addWeighted(img_sobelx, 0.5, img_sobely, 0.5, 0)    # combine sobel x and y

# laplacian
img_laplacian = cv2.Laplacian(img_gray, -1, ksize=3)

# print result
cv2.imshow("before", img)
cv2.waitKey()
cv2.imshow("after", img_laplacian)
cv2.waitKey()
cv2.destroyAllWindows()

# write result
cv2.imwrite("./img_processed/img_filter2D.jpg", img_filter2D)
cv2.imwrite("./img_processed/img_canny.jpg", img_canny)
cv2.imwrite("./img_processed/img_sobel.jpg", img_sobel)
cv2.imwrite("./img_processed/img_laplacian.jpg", img_laplacian)
