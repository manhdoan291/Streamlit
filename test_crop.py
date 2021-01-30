import cv2
import numpy as np
import os
image_path = 'dataset/test_data/cccd3.jpg'
image_input = cv2.imread(image_path)
def image_resize(image_input, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    # cropped_image.get_cropper()
    dim = None
    (h, w) = image_input.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image_input

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image_input, dim, interpolation = inter)

    # return the resized image
    return resized
    image = image_resize(image_input, width= 640, height = 720)
    crop_img1 = image[150:350, 30:190] # Crop from {x, y, w, h } => {0, 0, 300, 400}
    crop_img2 = image[350:500, 10:240]  #Crop from {x, y, w, h } => {0, 0, 300, 400}
    crop_img3 = image[150:550, 210:600] # Crop from {x, y, w, h } => {0, 0, 300, 400}
    crop_img4 = image[105:160, 210:570]

    cv2.imshow("testcrop1",crop_img1)
    cv2.imshow("testcrop2",crop_img2)
    cv2.imshow("testcrop3",crop_img3)
    cv2.imshow("testcrop4",crop_img4)
    cv2.waitKey(0)    