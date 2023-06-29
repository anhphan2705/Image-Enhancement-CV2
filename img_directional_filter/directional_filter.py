import cv2 as cv
import numpy as np

# img read
img = cv.imread("./img_data/moutain.jpg")

# kernel
kernel_0deg = 1/4 * np.array([[0, 0, 0],
                              [1, 2, 1],
                              [0, 0, 0]])
kernel_45deg = 1/4 * np.array([[0, 0, 1],
                              [0, 2, 0],
                              [1, 0, 0]])
kernel_90deg = 1/4 * np.array([[0, 1, 0],
                              [0, 2, 0],
                              [0, 1, 0]])
kernel_135deg = 1/4 * np.array([[1, 0, 0],
                              [0, 2, 0],
                              [0, 0, 1]])
# filter2D
img_0deg = cv.filter2D(img, ddepth=-1, kernel=kernel_0deg)
img_45deg = cv.filter2D(img, ddepth=-1, kernel=kernel_45deg)
img_90deg = cv.filter2D(img, ddepth=-1, kernel=kernel_90deg)
img_135deg = cv.filter2D(img, ddepth=-1, kernel=kernel_135deg)

# print result
cv.imshow("before", img)
cv.waitKey()
cv.imshow("after", img_135deg)
cv.waitKey()
cv.destroyAllWindows()

# write result
# cv.imwrite("./img_processed/img_0deg.jpg", img_0deg)
# cv.imwrite("./img_processed/img_45deg.jpg", img_45deg)
# cv.imwrite("./img_processed/img_90deg.jpg", img_90deg)
# cv.imwrite("./img_processed/img_135deg.jpg", img_135deg)
