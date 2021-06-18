#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

#reading in an image
image = mpimg.imread('images/done/solidWhiteRight.jpg')

#printing out some stats and plotting
#plt.imshow(gray, cmap='gray')

#plt.imshow(image) 

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def create_vertices(img):
    """
    'img' is a canny transform edge image
    
    Adjust our vertices here to be a trapezoid
    The top of the trapezoid should be where we first detect edges from the center looking bottom-up
    Sides of the trapezoid should extend to edges (plus buffer)
    """
    
    ysize, xsize = img.shape[0], img.shape[1]
    bottom_ignore = ysize//6
    ybuffer = ysize//30
    xbuffer_top = xsize//50
    xbuffer_bot = xbuffer_top*2
    side_search_buffer = ybuffer//2
     # Let's find the last white pixel's index in the center column.
    # This will give us an idea of where our region should be
    # We ignore a certain portion of the bottom of the screen so we get a better region top
    #   - This is partly because car hoods can obsure the region
    center_white = img[:ysize-bottom_ignore, xsize//2] == 255
    indices = np.arange(0, center_white.shape[0])
    indices[~center_white] = 0
    last_white_ind = np.amax(indices)
    
    # If our first white pixel is too close to the bottom of the screen, default back to the screen center
    # region_top_y = (last_white_ind if last_white_ind < 4*ysize//5 else ysize//2) + ybuffer
    region_top_y = min(last_white_ind + ybuffer, ysize-1)
    
    # Now we need to find the x-indices for the top segment of our region
    # To do this we will look left and right from our center point until we find white
    y_slice_top = max(region_top_y - side_search_buffer, 0)
    y_slice_bot = min(region_top_y + side_search_buffer, ysize-1)
    region_top_white = np.copy(img[y_slice_top:y_slice_bot, :]) == 255
    
    indices = np.zeros_like(region_top_white, dtype='int32')
    indices[:, :] = np.arange(0, xsize)
    indices[~region_top_white] = 0
    # Separate into right and left sides we can grab our indices easier:
    # Right side min and left side max
    right_side = np.copy(indices)
    right_side[right_side < xsize//2] = xsize*2  # Large number because we will take min
    left_side = np.copy(indices)
    left_side[left_side > xsize//2] = 0
    
    region_top_x_left = max(np.amax(left_side) - xbuffer_top, 0)
    region_top_x_right = min(np.amin(right_side) + xbuffer_top, xsize)
    
    # Now we do the same thing for the bottom
    # Look left and right from the center until we hit white
    indices = np.arange(0, xsize)
    region_bot_white = img[ysize-bottom_ignore, :] == 255
    indices[~region_bot_white] = 0
    
    # Separate into right and left sides we can grab our indices easier:
    # Right side min and left side max
    right_side = np.copy(indices)
    right_side[right_side < xsize//2] = xsize*2  # Large number because we 
    left_side = np.copy(indices)
    left_side[left_side > xsize//2] = 0
    
    region_bot_x_left = max(np.amax(left_side) - xbuffer_bot, 0)
    region_bot_x_right = min(np.amin(right_side) + xbuffer_bot, xsize)
    
    # Because of our bottom_ignore, we need to extrapolate these bottom x coords to bot of screen
    left_slope = ((ysize-bottom_ignore) - region_top_y)/(region_bot_x_left - region_top_x_left)
    right_slope = ((ysize-bottom_ignore) - region_top_y)/(region_bot_x_right - region_top_x_right)
    # Let's check these slopes we don't divide by 0 or inf
    if abs(left_slope < .001):
        left_slope = .001 if left_slope > 0 else -.001
    if abs(right_slope < .001):
        right_slope = .001 if right_slope > 0 else -.001
    if abs(left_slope) > 1000:
        left_slope = 1000 if left_slope > 0 else -1000
    if abs(right_slope) > 1000:
        right_slope = 1000 if right_slope > 0 else -1000
    # b=y-mx
    left_b = region_top_y - left_slope*region_top_x_left
    right_b = region_top_y - right_slope*region_top_x_right
    # x=(y-b)/m
    region_bot_x_left = max(int((ysize-1-left_b)/left_slope), 0)
    region_bot_x_right = min(int((ysize-1-right_b)/right_slope), xsize-1)
    
    
    verts = [
        (region_bot_x_left, ysize),
        (region_top_x_left, region_top_y),
        (region_top_x_right, region_top_y),
        (region_bot_x_right, ysize)
    ]
    
    return np.array([verts], dtype=np.int32)

def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    verts = create_vertices(img)
    cv2.fillPoly(mask, verts, ignore_mask_color)
    
    #Let's return an image of the regioned area in lines
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.polylines(line_img, verts, isClosed=True, color=[0, 255, 0], thickness=5)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    return masked_image, line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if lines is None: return lines
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    avg_lines = average_lines(lines, img)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     draw_lines(line_img, lines)
    draw_lines(line_img, avg_lines, color=[138,43,226])
    return line_img

def average_lines(lines, img):
    '''
    img should be a regioned canny output
    '''
    if lines is None: return lines
    
    positive_slopes = []
    positive_xs = []
    positive_ys = []
    negative_slopes = []
    negative_xs = []
    negative_ys = []
    
    min_slope = .3
    max_slope = 1000
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            
            if abs(slope) < min_slope or abs(slope) > max_slope: continue  # Filter our slopes
                
            # We only need one point sample and the slope to determine the line equation
            positive_slopes.append(slope) if slope > 0 else negative_slopes.append(slope)
            positive_xs.append(x1) if slope > 0 else negative_xs.append(x1)
            positive_ys.append(y1) if slope > 0 else negative_ys.append(y1)
    
    # We need to calculate our region_top_y from the canny image so we know where to extend our lines to
    ysize, xsize = img.shape[0], img.shape[1]
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    white = img == 255
    YY[~white] = ysize*2  # Large number because we will take the min
    region_top_y = np.amin(YY)
    
    new_lines = []
    if len(positive_slopes) > 0:
        m = np.mean(positive_slopes)
        avg_x = np.mean(positive_xs)
        avg_y = np.mean(positive_ys)
        
        b = avg_y - m*avg_x
        
        # We have m and b, so with a y we can get x = (y-b)/m
        x1 = int((region_top_y - b)/m)
        x2 = int((ysize - b)/m)
        
        new_lines.append([(x1, region_top_y, x2, ysize)])
    
    if len(negative_slopes) > 0:
        m = np.mean(negative_slopes)
        avg_x = np.mean(negative_xs)
        avg_y = np.mean(negative_ys)
        
        b = avg_y - m*avg_x
        
        # We have m and b, so with a y we can get x = (y-b)/m
        x1 = int((region_top_y - b)/m)
        x2 = int((ysize - b)/m)
        
        new_lines.append([(x1, region_top_y, x2, ysize)])
            
    return np.array(new_lines)

def weighted_img(initial_img, img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, l)

def save_img(img, name):
    mpimg.imsave('./images/output/{0}'.format(name if '.' in name else '{0}.png'.format(name)), img)

import os
image_names = [name for name in os.listdir("./images") if '.' in name]
image_names.sort()
# print(image_names)
images = [mpimg.imread('./images/{0}'.format(name)) for name in image_names]

def detect_lines(img, debug=False):
    ysize, xsize = img.shape[0], img.shape[1]
    
    blur_gray = gaussian_blur(grayscale(img), kernel_size=5)
    
    ht = 150  # First detect gradients above. Then keep between low and high if connected to high
    lt = ht//3  # Leave out gradients below
    canny_edges = canny(blur_gray, low_threshold=lt, high_threshold=ht)
    if debug: save_img(canny_edges, 'canny_edges_{0}'.format(index))
    
    # Our region of interest will be dynamically decided on a per-image basis 
    regioned_edges, region_lines = region_of_interest(canny_edges)

    rho = 2
    theta = 3*np.pi/180
    min_line_length = xsize//16
    max_line_gap = min_line_length//2
    threshold = min_line_length//4
    lines = hough_lines(regioned_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Let's combine the hough-lines with the canny_edges to see how we did
    overlayed_lines = weighted_img(img, lines)
    # overlayed_lines = weighted_img(weighted_img(img, region_lines, a=1), lines)
    if debug: save_img(overlayed_lines, 'overlayed_lines_{0}'.format(index))
    
    return overlayed_lines

for index, img in enumerate(images):
    print('Image:', index)
#     debug = (True if index == 0 else False)
    debug = False
    detect_lines(img, debug)

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    return detect_lines(image)

from moviepy.editor import VideoFileClip

#white_output = './videos/output/Output.mp4'
#clip1 = VideoFileClip("./videos/Short-daytime-test.mp4")
#white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
#white_clip.write_videofile(white_output, audio=False)

capture = cv2.VideoCapture("Test/test_videos/solidWhiteRight.mp4")
window_name="challenge"
if not capture.isOpened():
    exit(0)
    
while True:
    ret,img = capture.read()
    if img is None:
        break

    img =detect_lines(img, debug=False)
    
    cv2.imshow(window_name,img)

    k = cv2.waitKey(30)
    if k == ord("q"):
        break
    
capture.release()
cv2.destroyAllWindows()
cv2.waitKey(1)


