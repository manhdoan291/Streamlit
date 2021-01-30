import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import docx2txt
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber
import cv2
import numpy as np
import argparse
import glob
import os
from PIL import Image 
import pytesseract as pt 
from math import atan
from tqdm import tqdm
#Process_OCR_CCCD
class Cropper_class:
    def cropper(image_input):          
        # Part 1: Extracting only olored regions from the image
        #chuyen anh thanh gray
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        #lam mo anh bang ma tran 5 x 5
        blurred = cv2.GaussianBlur(image_input, (3,3),0)
        #tao mot adaptive thresh co gia tri 178 voi method la thresh mean c
        #tao kernel lay hinh vuong
        thresh = cv2.adaptiveThreshold(gray,178,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        #dan no anh lay theo pixel da so
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
        #Find the index of the largest contour
        (_, cnts, _) = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        for cnt in cnts:
            cnt = cnts[max_index]
            x,y,w,h = cv2.boundingRect(cnt)
        cropped_image = image_input[y:y+h, x:x+w]      
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        return cropped_image
class Detector_class(Cropper_class):
    def detector(cropped_image):
        def image_resize(cropped_image, width = None, height = None, inter = cv2.INTER_AREA):
            # initialize the dimensions of the image to be resized and
            # grab the image size
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
        # area_img1 = cv2.rectangle(image,(150,300),(350,190),(0,0,255),2)
        # area_img2 = cv2.rectangle(image,(350,10),(500,240),(0,0,255),2)
        # area_img3 = cv2.rectangle(image,(150,210),(550,600),(0,0,255),2)
        # area_img4 = cv2.rectangle(image,(105,210),(160,570),(0,0,255),2)
        # selectarea_image = [area_img1, area_img2, area_img3, area_img4]

        crop_img1 = image[150:350, 30:190] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        crop_img2 = image[360:420, 10:240]  #Crop from {x, y, w, h } => {0, 0, 300, 400}
        crop_img3 = image[150:370, 210:630] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        crop_img4 = image[105:150, 210:570]
        detected_images = [crop_img1, crop_img2, crop_img3, crop_img4]
        return detected_images
class Ocr_class(Detector_class):
    def ocr(detected_images):
        fullTempPath ="./output_ocr.txt"  #de fulltemppTH DE LEN THANH BIEN CUA HAM
        text2 = pt.image_to_string(detected_images[1], lang ="vie")
        print("\n")
        text3 = pt.image_to_string(detected_images[2], lang ="vie") 
        print("\n")
        text4 = pt.image_to_string(detected_images[3], lang ="vie")  
        text_images = [text2,text3,text4]
        return text_images

# MAIN
image_path = 'dataset/test_data/cccd3.jpg'
image_input = cv2.imread(image_path)
Cropper_class.cropper(image_input)
cropped_image = Cropper_class.cropper(image_input)
Detector_class.detector(cropped_image)
detected_images = Detector_class.detector(cropped_image)
selectarea_image = Detector_class.detector(cropped_image)
Ocr_class.ocr(detected_images)
text_images = Ocr_class.ocr(detected_images)
# Process_Streamlit
def read_pdf(file):
	pdfReader = PdfFileReader(file)
	count = pdfReader.numPages
	all_page_text = ""
	for i in range(count):
		page = pdfReader.getPage(i)
		all_page_text += page.extractText()

	return all_page_text

def read_pdf_with_pdfplumber(file):
	with pdfplumber.open(file) as pdf:
	    page = pdf.pages[0]
	    return page.extract_text()

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img 

st.markdown('<h2>HỆ THỐNG NHẬN DẠNG ẢNH CĂN CƯỚC CÔNG DÂN</h2>',unsafe_allow_html=True)

def main():
    menu = ["Trang chu","Tap du lieu","File Document","Gioi thieu"]
    choice = st.sidebar.selectbox("Menu Left",menu) 
    if choice == "Trang chu":
        image_file = st.file_uploader("Tải ảnh căn cước công dân từ người dùng",type=['png','jpeg','jpg'])
        if image_file is not None:
            st.text("Thông tin ảnh đầu vào :")
            file_details = {"Tên file ảnh":image_file.name,"Loại file":image_file.type,"Kích thước file":image_file.size}
            st.write(file_details)
            img = load_image(image_file)
            st.image(img,width=250,height=250)
            if st.button("Xử lí cắt viền căn cước công dân :"):
                st.image(cropped_image,width=250,height=250)
            if st.button("Xử lí cắt phần ảnh chứa thông tin :"):
                st.image(detected_images,width=250,height=250)
            if st.button("Trích xuất thông tin trong ảnh :"):
                st.write(text_images)

    elif choice == "Tap du lieu":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if st.button("Xu li"):
            if data_file is not None:
                file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
                st.write(file_details)  
                df = pd.read_csv(data_file)                
                st.dataframe(df)

    elif choice == "File Document":
    	st.subheader("File Document")
    	docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
    	if st.button("Xu li"):
    		if docx_file is not None:
    			file_details = {"Ten file":docx_file.name,"Loai file":docx_file.type,"Kich thuoc file":docx_file.size}
    			st.write(file_details)
    			# Check File Type
    			if docx_file.type == "text/plain":
    				st.text(str(docx_file.read(),"utf-8")) # empty
    				raw_text = str(docx_file.read(),"utf-8") # works with st.text and st.write,used for futher processing
    				st.write(raw_text) # works
    			elif docx_file.type == "application/pdf":
    				try:
    					with pdfplumber.open(docx_file) as pdf:
    					    page = pdf.pages[0]
    					    st.write(page.extract_text())
    				except:
    					st.write("None")   
    			elif docx_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
    			# Use the right file processor ( Docx,Docx2Text,etc)
    				raw_text = docx2txt.process(docx_file) # Parse in the uploadFile Class directory
    				st.write(raw_text)  
    else:
    	st.subheader("Giới thiệu")
    	st.info("B Streamlit")
    	st.info("Jesus Saves @JCharisTech")
    	st.text("Jesse E.Agbe(JCharis)")
if __name__ == '__main__':
	main()
components.html("""
<!DOCTYPE html>
<html>
<head>
    <meta charset=" utf-8">
    <meta name=" viewport" content=" width=device-width, initial-scale=1">
    <link rel=" stylesheet" href=" https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <script src=" https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
    <script src=" https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js"></script>
    <script src=" https://maxcdn.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
</head>
<body >
    <div class="container" style ="margin-right: 880px">
    <div class="row">
        <div class="col-lg-4 col-sm-6 portfolio-item" style="margin-top:20px;">
            <div class="card h-100">
                <a href="#"> <img src="http://placehold.it/700x400" class="card-img-top" /></a>
                <div class="card-body">
                    <h4 class="card-title">
                        <a href="#">Project One</a>
                    </h4>
                    <p class="card-text">
                        Lorem ipsum dolor sit amet, consectetur adipisicing elit.
                        Amet numquam aspernatur eum quasi sapiente nesciunt? Voluptatibus sit,
                        repellat sequi itaque deserunt, dolores in, nesciunt, illum tempora ex quae?
                        Nihil, dolorem!
                    </p>
                </div>
            </div>
        </div>
        <div class="col-lg-4 col-sm-6 portfolio-item" style="margin-top:20px;">
            <div class="card h-100">
                <a href="#"> <img src="http://placehold.it/700x400" class="card-img-top" /></a>
                <div class="card-body">
                    <h4 class="card-title">
                        <a href="#">Project Two</a>
                    </h4>
                    <p class="card-text">
                        Lorem ipsum dolor sit amet, consectetur adipiscing elit.
                        Nam viverra euismod odio, gravida pellentesque urna varius vitae.
                    </p>
                </div>
            </div>
        </div>
        <div class="col-lg-4 col-sm-6 portfolio-item" style="margin-top:20px;">
            <div class="card h-100">
                <a href="#"> <img src="http://placehold.it/700x400" class="card-img-top" /></a>
                <div class="card-body">
                    <h4 class="card-title">
                        <a href="#">Project Three</a>
                    </h4>
                    <p class="card-text">
                        Lorem ipsum dolor sit amet, consectetur adipisicing elit.
                        Quos quisquam, error quod sed cumque, odio distinctio velit nostrum temporibus
                        necessitatibus et facere atque iure perspiciatis mollitia recusandae vero vel quam!
                    </p>
                </div>
            </div>
        
        </div>
     
    </div>
    <div class="row" style="margin-top:20px;">
        <div class="col-lg-6">
            <h2> Modern Business Features</h2>
            <p>The Modern Business template by Start Bootstrap includes:</p>
            <ul>
                <li><b>Bootstrap v4</b></li>
                <li>jQuery</li>
                <li>Font Awesome</li>
                <li>Working contact form with validation</li>
                <li>Unstyled page elements for easy customization</li>
            </ul>
            <p>
                Lorem ipsum dolor sit amet, consectetur adipisicing elit. Corporis, 
                omnis doloremque non cum id reprehenderit, 
                quisquam totam aspernatur tempora minima unde aliquid ea culpa sunt.
                Reiciendis quia dolorum ducimus unde.
            </p>

        </div>
        <div class="col-lg-6">
            <img src="http://placehold.it/700x450" class="img-fluid rounded" />
        </div>
    </div>
        <hr />
        <div class="row mb-4">
            <div class="col-md-8">
                <p>Lorem ipsum dolor sit amet, consectetur adipisicing elit. Molestias, expedita, saepe,
                vero rerum deleniti beatae veniam harum neque 
                nemo praesentium cum alias asperiores commodi.</p>
            </div>
            <div class="col-md-4">
                <a class="btn btn-lg btn-secondary btn-block" href="#">Call to Action</a>
            </div>
        </div>

        </div>
    <footer class="py-5 bg-dark" style="width:948px">
        <div class="container">
            <p class="m-0 text-center text-white">Copyright © Your Website 2020</p>
        </div>
    </footer>


</body>
</html>
       
    """,
    width=1000,
    height=1500,
)