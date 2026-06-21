import cv2

img = cv2.imread("test_images/image12.jpg")

cv2.imshow("Test Window", img)

cv2.waitKey(0)

cv2.destroyAllWindows()