import streamlit as st
# from streamlit_back_camera_input import back_camera_input
import cv2
import pickle
import numpy as np
from object_detector import *
# from scipy.spatial.distance import euclidean
import csv
from PIL import Image
import firebase_admin
from firebase_admin import db,credentials
import warnings
warnings.filterwarnings("ignore")

#Authentication to Firebase
try:
    app = firebase_admin.get_app()
except ValueError as e:
    cred = credentials.Certificate("cropdata.json")
    firebase_admin.initialize_app(cred, {"databaseURL": "https://cropdata-3413b-default-rtdb.firebaseio.com/"})

st.set_page_config(page_title="Maize Crop Yield Estimation", page_icon="🌱")

def stat():
    try:
        st.subheader("Weight Prediction using ML Model for Statistical Data")
        a=float(st.number_input("Height in cm"))
        b=float(st.number_input("Width in cm"))
        plant=st.number_input("Enter the no of Plants")

        with open(r"RegModel.pkl", 'rb') as file:
            regressor = pickle.load(file)

        #Inputs
        height=[[a,b]]
        pred=(round(regressor.predict(height)[0],2))

        if a<=6 or b<=5:
            st.error("Enter proper Inputs")
        else:
            btn=st.button("Predict")
            if btn:
                st.text("Weight Predicted in g : ")
                st.text(pred)
                calc=pred*2*plant
                z= (calc)/1000 
                st.text("Total Weight in kg: ")
                st.text(z)
                #Firebase 
                fire={"Length":a,"Width":b,"Weight":pred,"plantcount":plant,"calculation":calc}
                db.reference("/StatData").push().set(fire)

                #Storage
                SData = [a, b, pred,plant,calc]
                with open('StatData.csv','a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(SData)
                # st.balloons()   
                st.success("Successfully Saved")
    except:
        st.warning("Enter proper Inputs")

def image():
    try:
        # Load Aruco detector
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters =  cv2.aruco.DetectorParameters()
        # Load Object Detector
        detector = HomogeneousBgDetector()

        st.subheader("Weight Prediction using ML Model for Image Data")
        imageinput = st.file_uploader("",type=['png', 'jpg'])
        plant=st.number_input("Enter the no of Plants")
        
        btn = st.button("Predict")

        if btn:
            # Load Image
            img = cv2.imdecode(np.fromstring(imageinput.read(), np.uint8),1)

            # Get Aruco marker
            corners, _, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

            # Draw polygon around the marker
            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(corners[0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 20

            contours = detector.detect_objects(img)

            # Draw objects boundaries
            for cnt in contours:
                # Get rect
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect

                # Get Width and Height of the Objects by applying the Ratio pixel to cm
                object_width = w / pixel_cm_ratio
                object_height = h / pixel_cm_ratio

                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.polylines(img, [box], True, (255, 0, 0), 2)
                cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            
            I1 = round(object_height,3)
            I2 = round(object_width,3)

        # Connecting to ML Model
            with open(r"RegModel.pkl", 'rb') as file:
                regressor = pickle.load(file)

        #inputs
            height=[[I1,I2]]
            pred=(round(regressor.predict(height)[0],3))

        #Display
            st.image(img)
            st.text("Height Predicted in cm : ")
            st.text(I1)
            st.text("Width Predicted in cm : ")
            st.text(I2)
            st.text("Weight Predicted in gm : ")
            st.text(pred)
            calc=pred*2*plant
            z= (calc)/1000 
            st.text("Total Weight in kg: ")
            st.text(z)

            #storage in firebase
            f={"Length":I1,"Width":I2,"Weight":pred,"Plantcount":plant,"Calculation":calc}
            i=0
            db.reference("/ImageData").push(i+1).set(f)

            #Storage in csv
            fields = [I1, I2, pred,plant,calc]
            
            with open('Database.csv','a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
            
            st.success("Successfully Saved")
    
    except IndexError:
        st.warning("Select the Valid image with Proper ArUco Marking")
    except AttributeError:
        st.warning("Select the image before Predicting")
    
def capture():
    st.subheader("Weight Prediction using ML Model for Capturing Image")
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters =  cv2.aruco.DetectorParameters()
        # Load Object Detector
        detector = HomogeneousBgDetector()

        plant=st.number_input("Enter the no of Plants")
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            # To read image file buffer with OpenCV:
            bytes_data = img_file_buffer.getvalue()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            # Get Aruco marker
            corners, _, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)

            # Draw polygon around the marker
            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

            # Aruco Perimeter
            aruco_perimeter = cv2.arcLength(corners[0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 20

            contours = detector.detect_objects(img)

            # Draw objects boundaries
            for cnt in contours:
                # Get rect
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect

                # Get Width and Height of the Objects by applying the Ratio pixel to cm
                object_width = w / pixel_cm_ratio
                object_height = h / pixel_cm_ratio

                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.polylines(img, [box], True, (255, 0, 0), 2)
                cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            
            I1 = round(object_height,3)
            I2 = round(object_width,3)

        # Connecting to ML Model
            with open(r"RegModel.pkl", 'rb') as file:
                regressor = pickle.load(file)

        #inputs
            height=[[I1,I2]]
            pred=(round(regressor.predict(height)[0],3))

        #Display
            st.image(img)
            st.text("Height Predicted in cm : ")
            st.text(I1)
            st.text("Width Predicted in cm : ")
            st.text(I2)
            st.text("Weight Predicted in gm: ")
            st.text(pred)
            calc=pred*2*plant
            z= (calc)/1000 
            st.text("Total Weight in kg: ")
            st.text(z)

            #storage in firebase
            f={"Length":I1,"Width":I2,"Weight":pred,"Plantcount":plant,"Calculation":calc}
            db.reference("/capture").push().set(f)

            #Storage in csv
            fields = [I1, I2, pred,plant,calc]
            
            with open('CaptureData.csv','a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

            st.success("Successfully Saved")   
    except IndexError:
        st.error("Check for ArUco Marker while Capturing")

def imagedata():
    st.title("ImageData Database")
    x=db.reference("/ImageData").get()
    st.table(x)

def statdata():
    st.title("Statistical Data Database")
    y=db.reference("/StatData").get()
    st.table(y)

def capturedata():
    st.title("Captured Data Database")
    z=db.reference("/capture").get()
    st.table(z)
      
def Home():
    
    # col1,col2,col3= st.columns(3)

    # with col1:
    #     st.image("download.jpeg")

    # with col2:
    #     st.image("download.png",width=150)

    # with col3:
    #     st.image("unnamed.png",width=100)

    st.title("ML Model for Crop Yield Estimation")
    st.subheader("Find the Weight of Maize Crop in 3 Methods")
    st.subheader("1. Statistical Data")
    st.text("This application allows the user to input the height and width of a crop and uses \na pre-trained machine learning model to predict the weight of the crop.")
    st.subheader("2.Image Data")
    st.text("This application allows the user to upload the image and predict the height \nand width of a crop and uses a pre-trained machine learning model to predict \nthe weight of the crop.")
    st.subheader("3.Capturing the Image")
    st.text("This application allows the user to capture the image and predict the height \nand width of a crop and uses a pre-trained machine learning model to predict \nthe weight of the crop.")
    
def main():
    # st.markdown("<h1 style='text-align: center; color: red;'>Big headline</h1>", unsafe_allow_html=True)
    # col1,col2,col3= st.columns(3)

    # with col2:
    # st.sidebar.image("download.jpeg")
    # with col3:
    #     st.sidebar.image("download.png",width=100)

    selected_box = st.sidebar.selectbox('Choose one of the following',('Home','Statistical Data','Image Data','Capture Image','View Statistical Data','View Image Data','View Capture Data'))
    if selected_box == 'Home':
        Home()
    if selected_box == 'Statistical Data':
        stat() 
    if selected_box == 'Image Data':
        image()
    if selected_box == 'Capture Image':
        capture()
    if selected_box == 'View Image Data':
        imagedata()
    if selected_box == 'View Statistical Data':
        statdata()
    if selected_box == 'View Capture Data':
        capturedata()
    
if __name__ == "__main__":
    main()