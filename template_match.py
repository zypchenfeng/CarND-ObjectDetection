import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('./cutouts/bbox-example-image.jpg')
# image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['./cutouts/cutout1.jpg', './cutouts/cutout2.jpg', './cutouts/cutout3.jpg',
            './cutouts/cutout4.jpg', './cutouts/cutout5.jpg', './cutouts/cutout6.jpg']


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image and a list of templates as inputs
# then searches the image and returns the a list of bounding boxes
# for matched templates
def find_matches(img, template_list):
    # Make a copy of the image to draw on
    imcopy = np.copy(img)
    # Define an empty list to take bbox coords
    bbox_list = []
    # Iterate through template list
    for temp_id in template_list:
        # Read in templates one by one
        temp_img = mpimg.imread(temp_id)
        w, h = temp_img.shape[1], temp_img.shape[0]
        # Use cv2.matchTemplate() to search the image
        #     using whichever of the OpenCV search methods you prefer
        result = cv2.matchTemplate(img, temp_img, cv2.TM_CCOEFF)
        # Use cv2.minMaxLoc() to extract the location of the best match.
        bbox_start = cv2.minMaxLoc(result)[3] # I used correlation coefficient, so looking at the maximum result/location
        bbox_list.append((bbox_start, (bbox_start[0] + w, bbox_start[1] + h)))
        # pass
    # Determine bounding box corners for the match
    # Return the list of bounding boxes
    return bbox_list


bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()
