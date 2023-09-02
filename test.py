import pickle
import numpy as np 
import tensorflow
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2

feature_list = np.array(pickle.load(open('embeddings.pkl' , 'rb'))) # rb = read binary

#print(np.array(feature_list).shape) #feature list is a 2D array which has 44441 images(row) and in each row 2048 features 

filenames = pickle.load(open('filenames.pkl' , 'rb'))

model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

#now import a sample img
img = image.load_img('sample/34981.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array,axis=0)
preprocess_img = preprocess_input(expanded_img_array)
result = model.predict(preprocess_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6 , algorithm='brute',metric='euclidean') # 6 neighbor isliye kyuki 1 to vo img khud ko hi recommend krr degi , or baki ke 5 nearest vale
neighbors.fit(feature_list)                                                        # brute isliye  kyuki data jayada bda nhi h only 44441 img hi h
                                                                                   # we can also use cosin ditance but eucliden is giving good results so we'll use this 

distances , indices = neighbors.kneighbors([normalized_result]) #normalized_result vector ke 5 nearest vector nikale h feature_list me se euclidean ke base pe

print(indices)

#now print those 5 images
# for file in indices[0]:
#     print(filenames[file])

#now displaying the images
for file in indices[0][1:6]: # 0'th index ko isliye chod diya kyuki vo khud input img hi h isliye 1 to 6 
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output' , cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)