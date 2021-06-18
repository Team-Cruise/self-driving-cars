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
        plt.imshow(img[:,:,::-1]);
        # plt.imshow(img);
    plt.show()

street = cv2.imread('Test/test_images/solidWhiteRight.jpg')
street_gray = cv2.imread('Test/test_images/solidWhiteRight.jpg',0)


# imshow(street)

street_rgb = cv2.cvtColor(street,cv2.COLOR_BGR2RGB)
street_gray = cv2.cvtColor(street_rgb,cv2.COLOR_RGB2GRAY)

# imshow(street_gray, False, False)

pts = np.array([[
    [100, 550],
    [930, 550],
    [510, 315],
    [440, 315],
]])

pts = pts.reshape((-1, 1, 2))

street_copy = street_gray.copy()
street_copy = cv2.cvtColor(street_rgb, cv2.COLOR_BGR2RGB)

color = (255, 0, 255)
cv2.polylines(street_copy, pts, True, color, 4)

# imshow(street_copy)

mask = np.zeros(street_copy.shape[:3], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

# imshow(mask)


## (3) do bit-op
# dst = cv2.bitwise_and(mask, mask, mask=mask)

# imshow(dst)
blur = cv2.GaussianBlur(street_copy,(5,5),0)
# imshow(blur)

ret, thr =cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)
# imshow(thr)

canny = cv2.Canny(thr,100,200)

# imshow(canny, False, False)




lines = cv2.HoughLines(mask,1,np.pi/180,200)

img_copy = street.copy()
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img_copy,(x1,y1),(x2,y2),(0,255,0),2)

imshow(img_copy)


# imshow(dst)


# imshow(street_rgb)
# plt.imshow(street_gray, cmap='gray')
# imshow(street_gray)
# imshow(thr)
# plt.imshow(thr, cmap='gray')
# imshow(canny)

# imshow(lines, False, True)

