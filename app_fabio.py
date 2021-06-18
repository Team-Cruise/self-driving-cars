import cv2
import numpy as np
import matplotlib.pyplot as plt

print(cv2.__version__)

def imshow(img, enlarge = True, color = True):
    if enlarge:
        plt.figure(figsize=(15,10));
    if not color:
        plt.imshow(img, cmap='gray');
    else:
        # plt.imshow(img[:,:,::-1]);
        plt.imshow(img);
    plt.show()

street = cv2.imread('Test/test_images/solidWhiteRight.jpg')
street_gray = cv2.imread('Test/test_images/solidWhiteRight.jpg',0)


# imshow(street)

street_rgb = cv2.cvtColor(street,cv2.COLOR_BGR2RGB)
street_gray = cv2.cvtColor(street_rgb,cv2.COLOR_BGR2GRAY)



points = np.array([[
    [100, 550],
    [930, 550],
    [510, 315],
    [440, 315],
]])

street_copy = street.copy()
street_copy = cv2.cvtColor(street, cv2.COLOR_BGR2RGB)

color = (255, 0, 255)
cv2.polylines(street_copy, points, True, color, 4)


ret, thr =cv2.threshold(street_gray,127,255,cv2.THRESH_BINARY_INV)

canny = cv2.Canny(street_gray,100,200)

lines = cv2.HoughLines(canny,1,np.pi/180,200)



# imshow(street_copy)

# imshow(street_rgb)
# plt.imshow(street_gray, cmap='gray')
# imshow(street_gray)
# imshow(thr)
# plt.imshow(thr, cmap='gray')
# imshow(canny)

plt.imshow(lines)
