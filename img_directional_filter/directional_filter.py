import cv2 as cv
import numpy as np

# img read
img = cv.imread("./img_data/RawData.jpg")

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

# filter2D
img_0deg = cv.filter2D(img, ddepth=-1, kernel=kernel_0deg)
img_45deg = cv.filter2D(img, ddepth=-1, kernel=kernel_45deg)
img_90deg = cv.filter2D(img, ddepth=-1, kernel=kernel_90deg)

# print result
cv.imshow("before", img)
cv.waitKey()
cv.imshow("after", img_90deg)
cv.waitKey()
cv.destroyAllWindows()