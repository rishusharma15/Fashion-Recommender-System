import streamlit as st
import os
from PIL import Image  #PIL = python imaging library
import numpy as np 
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

st.title('Fashion Recommender System')

feature_list = np.array(pickle.load(open('embeddings.pkl' , 'rb')))
filenames = pickle.load(open('filenames.pkl' , 'rb'))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


# 1.file upload then save it
def save_uploaded_file(uploaded_file): #function to save files
    try:
        with open(os.path.join('uploads' , uploaded_file.name ), 'wb') as f: #we open a file 
            f.write(uploaded_file.getbuffer()) #after opening a file we write the content of it (buffer) into that file opend filr
        return 1
    except:
        return 0


def feature_extraction(img_path , model):   #function to extract features
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array,axis=0)
    preprocess_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocess_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(feature,feature_list): #fuction for recommendation
    neighbors = NearestNeighbors(n_neighbors=6 , algorithm='brute',metric='euclidean')
    neighbors.fit(feature_list)  
    distances , indices = neighbors.kneighbors([feature])

    return indices

uploaded_file = st.file_uploader("choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # 2.load file then feature extract
        #display the file->
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        #now feature extraction 
        feature = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(feature) #this line was to print the feature text
        
        # 3.recommendetions
        indices = recommend(feature , feature_list)

        # 4.show recommendetions
        col1,col2,col3,col4,col5, = st.columns(5)

        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

    else:
        st.header("some error occured in file upload")


