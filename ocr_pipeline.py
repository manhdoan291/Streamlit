import cv2
import numpy as np
import numpy 
import argparse
import glob
import os
from PIL import Image 
import pytesseract as pt 
from math import atan
from tqdm import tqdm
class Cropper_class:
    def __init__(self,image):
        self.image = image
    def cropper(self):
        def calculate_area(w, h):
            return w * h     
        origin_h, origin_w = image.shape[:2]
            
        # Part 1: Extracting only olored regions from the image
        #chuyen anh thanh gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #lam mo anh bang ma tran 5 x 5
        blurred = cv2.GaussianBlur(image, (3,3),0)
        #tao mot adaptive thresh co gia tri 178 voi method la thresh mean c
        #tao kernel lay hinh vuong
        thresh = cv2.adaptiveThreshold(gray,178,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        #dan no anh lay theo pixel da so
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
        #ap dung kernel vao anh de lay cac doan hinh vuong
        #Find the index of the largest contour
        (_, cnts, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        for cnt in cnts:
            cnt = cnts[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
        cropped_image = image[y:y+h, x:x+w]         
    
        # _bbox_area = calculate_area(w, h)
        # _image_area = calculate_area(origin_w,origin_h)
    
        # ratio = _bbox_area / _image_area
        # image = cv2.rectangle(image, (x, y), (x + w, y + h),(255, 0, 0), 2)
    
        # if x <4 and x>90:
        #     cv2.imwrite(os.path.join(wrong_fol, image_name), image)
        #     cv2.imwrite(os.path.join(cut_wrong_fol, image_name), cropped)
                
        # if y < 4 and x>90:
        #     cv2.imwrite(os.path.join(cut_wrong_fol, image_name), cropped)
        #     cv2.imwrite(os.path.join(wrong_fol, image_name), image)
    
        # if ratio > 0.2 and ratio < 0.9:
        #     cv2.imwrite(os.path.join(cut_save_fol, image_name), cropped)
        #     cv2.imwrite(os.path.join(save_fol, image_name), image)
        # else:
        #     cv2.imwrite(os.path.join(wrong_fol, image_name), image)
        #     cv2.imwrite(os.path.join(cut_wrong_fol, image_name), cropped)
    
        return cropped_image
class Detector_class(Cropper_class):
    def detector(cropped_image):
        def image_resize(cropped_image, width = None, height = None, inter = cv2.INTER_AREA):
            # initialize the dimensions of the image to be resized and
            # grab the image size
            # cropped_image.get_cropper()
            dim = None
            (h, w) = cropped_image.shape[:2]
        
            # if both the width and height are None, then return the
            # original image
            if width is None and height is None:
                return cropped_image
        
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
            resized = cv2.resize(cropped_image, dim, interpolation = inter)
        
            # return the resized image
            return resized
        image = image_resize(cropped_image, width= 640, height = 720)
        crop_img1 = image[150:350, 30:190] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        crop_img2 = image[360:420, 10:240]  #Crop from {x, y, w, h } => {0, 0, 300, 400}
        crop_img3 = image[150:550, 225:620] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        crop_img4 = image[105:150, 210:570]
        detected_images = [crop_img1, crop_img2, crop_img3, crop_img4]
        return detected_images
class Ocr_class(Detector_class):
    def ocr(detected_images):
        fullTempPath ="./output_ocr.txt"  #de fulltemppTH DE LEN THANH BIEN CUA HAM
        # iterating the images inside the folder 
        # applying ocr using pytesseract for python 
        # text1 = pt.image_to_string(detected_images[0], lang ="vie") 
        text2 = pt.image_to_string(detected_images[1], lang ="vie")
        text3 = pt.image_to_string(detected_images[2], lang ="vie") 
        text4 = pt.image_to_string(detected_images[3], lang ="vie")  
        # saving the  text for appending it to the output.txt file 
        # a + parameter used for creating the file if not present 
        # and if present then append the text content 
        file1 = open(fullTempPath, "a+") 
        # providing the content in the image 
        # file1.write(text1+"\n")
        file1.write(text2+"\n") 
        file1.write(text3+"\n") 
        file1.write(text4+"\n") 
     
        file1.close()  
    
        # for printing the output file 
        file2 = open(fullTempPath, 'r') 
        ocr_image = print(file2.read()) 
        file2.close()
        return  ocr_image
# Main - XU LY TAO HAM DÈF MAIN 
image_path = 'dataset/test_data/cccd3.jpg'
image = cv2.imread(image_path)
Cropper_class.cropper(image)
cropped_image = Cropper_class.cropper(image)
Detector_class.detector(cropped_image)
detected_images = Detector_class.detector(cropped_image)
Ocr_class.ocr(detected_images)