import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from PIL import Image 
from PyPDF2 import PdfFileReader
import pdfplumber
import cv2
import numpy as np
import glob
import os
from PIL import Image 
import pytesseract as pt 
from math import atan
from tqdm import tqdm
import SessionState
import awesome_streamlit as ast
from awesome_streamlit.core.services import resources
#Process_OCR_CCCD
class Cropper_class:
    def cropper(image_input):          
        # Part 1: Extracting only olored regions from the image
        #Chuyen anh thanh gray
        gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
        #Lam mo anh bang ma tran 5 x 5
        blurred = cv2.GaussianBlur(image_input, (3,3),0)
        #Tao mot adaptive thresh co gia tri 178 voi method la thresh mean c
        #Tao kernel lay hinh vuong
        thresh = cv2.adaptiveThreshold(gray,178,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4,4))
        #Dan no anh lay theo pixel da so
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
        #Anh chua mat cmtnd
        crop_img1 = image[150:350, 30:190] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        #Có giá trị đến
        crop_img2 = image[105:150, 210:570]  #Crop from {x, y, w, h } => {0, 0, 300, 400}
        #Ho va ten 220
        crop_img3 = image[140:200, 205:570] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        #Ngay thang nam sinh
        crop_img4 = image[195:235, 210:550] # Crop from {x, y, w, h } => {0, 0, 300, 400}      
        #Gioi tinh 
        crop_img5 = image[235:265, 210:350] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        #Quoc tich 
        crop_img6 = image[230:265, 360:620] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        #Que quan
        crop_img7 = image[265:315, 210:620] # Crop from {x, y, w, h } => {0, 0, 300, 400}
        #Noi thuong tru
        crop_img8 = image[315:370, 215:620]
        #Co gia tri den
        crop_img9 = image[360:420, 10:240]  #Crop from {x, y, w, h } => {0, 0, 300, 400}  
        detected_images = [crop_img1, crop_img2, crop_img3, crop_img4,crop_img5,crop_img6,crop_img7,crop_img8,crop_img9]
        return detected_images
class Ocr_class(Detector_class):
    def ocr(detected_images):
        text1 = detected_images[0]
        text2 = pt.image_to_string(detected_images[1], lang ="vie")
        text3 = pt.image_to_string(detected_images[2], lang ="vie") 
        text4 = pt.image_to_string(detected_images[3], lang ="vie")
        text5 = pt.image_to_string(detected_images[4], lang ="vie")
        text6 = pt.image_to_string(detected_images[5], lang ="vie")
        text7 = pt.image_to_string(detected_images[6], lang ="vie")
        text8 = pt.image_to_string(detected_images[7], lang ="vie")
        text9 = pt.image_to_string(detected_images[8], lang ="vie")
        text_images = (text1,text2,text3,text4,text5,text6,text7,text8,text9)
        return text_images
# MAIN
image_path = 'dataset/test_data/cccd100.jpg'
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
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

icon("search")
# selected = st.text_input("", "Search...")
tags = ast.shared.components.multiselect(
        "Search(s)", options=ast.database.TAGS, default=[]
    )
button_clicked = st.button("OK")

def main():
    st.sidebar.title("Navigation")
    menu = ["Trang chủ","Tập dữ liệu","File Document","Đăng nhập"]
    choice = st.sidebar.radio("Goto",menu) 
    if choice == "Trang chủ":
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
            session_state = SessionState.get(name="", button_sent=False)
            button_sent = st.button("Trích xuất thông tin trong ảnh")
            if button_sent:
                session_state.button_sent = True
            if session_state.button_sent:
                st.markdown("Avatar",unsafe_allow_html=True)
                st.image(detected_images[0],width=150,height=60)
                st.text_input(" Số chứng minh thư nhân dân:", text_images[1].replace("số:", ""))
                st.text_input(" Họ và tên:", text_images[2].replace("Họ và tên:", ""))
                st.text_input(" Ngày, tháng, năm sinh:", text_images[3].replace("Ngày, tháng, năm sinh:", ""))
                st.text_input(" Giới tính:", text_images[4].replace("Giới tính:", ""))
                st.text_input(" Quốc tịch:", text_images[5].replace("Quốc tịch:", ""))
                st.text_input(" Quê quán:", text_images[6].replace("Quê quán:", ""))
                st.text_input(" Nơi thường trú:", text_images[7].replace("Nơi thường trú:", ""))
                st.text_input(" Có giá trị đến:", text_images[8].replace("Có giá trị đến:", ""))
                st.button("Nộp thông tin")
                       
    elif choice == "Tập dữ liệu":
        st.subheader("Dataset")
        data_file = st.file_uploader("Upload CSV",type=['csv'])
        if st.button("Xử lí"):
            if data_file is not None:
                file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
                st.write(file_details)  
                df = pd.read_csv(data_file)                
                st.dataframe(df)

    elif choice == "File Document":
    	st.subheader("File Document")
    	docx_file = st.file_uploader("Upload File",type=['txt','docx','pdf'])
    	if st.button("Xử lí"):
    		if docx_file is not None:
    			file_details = {"Tên file":docx_file.name,"Loại file":docx_file.type,"Kích thước file":docx_file.size}
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
    st.sidebar.title("Contribute")
    st.sidebar.info(
        "This an open source project and you are very welcome to **contribute** your awesome "
        "comments, questions, resources and apps as "
        "[issues](https://github.com/MarcSkovMadsen/awesome-streamlit/issues) of or "
        "[pull requests](https://github.com/MarcSkovMadsen/awesome-streamlit/pulls) "
        "to the [source code](https://github.com/MarcSkovMadsen/awesome-streamlit). "
    )
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by Marc Skov Madsen. You can learn more about me at
        [datamodelsanalytics.com](https://datamodelsanalytics.com).""")

if __name__ == '__main__':
	main()
author_all = ast.shared.models.Author(name="All", url="")
author = st.selectbox("Select Author", options=[author_all] + ast.database.AUTHORS)
if author == author_all:
    author = None
show_awesome_resources_only = st.checkbox("Show Awesome Resources Only", value=True)
if not tags:
    st.info(
        """Please note that **we list each resource under a most important tag only!**"""
    )
resource_section = st.empty()
with st.spinner("Loading resources ..."):
    markdown = resources.get_resources_markdown(
        tags, author, show_awesome_resources_only
    )
    resource_section.markdown(markdown)
if st.sidebar.checkbox("Show Resource JSON"):
    st.subheader("Source JSON")
    st.write(ast.database.RESOURCES)
tags = None