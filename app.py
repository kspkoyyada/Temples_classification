import streamlit as st
from PIL import Image
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
import keras
import cv2

st.markdown('<h1 style="color:white;">Vgg 19 Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> street,  buildings, forest, sea, mountain, glacier</h3>', unsafe_allow_html=True)


upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
c1, c2= st.columns(2)
if upload is not None:
  im= Image.open(upload)
  im= im.resize((224,224))
  img= np.asarray(im)
    
  #image= cv2.resize(img,(224, 224,3))
  img= preprocess_input(img)
  img= np.expand_dims(img, 0)
  c1.header('Input Image')
  c1.image(im)
  c1.write(img.shape)

#load weights of the trained model.
  input_shape = (224, 224, 3)
  #optim_1 = Adam(learning_rate=0.0001)
  #n_classes=3
  #vgg_model = model(input_shape, n_classes, optim_1, fine_tune=2)
  res1 = keras.saving.load_model("./res_img_70_epochs.h5")
  
  # prediction on model
  res_preds = res1.predict(img)
  res_pred_classes = np.argmax(res_preds, axis=1)
  c2.header('Output')
  c2.subheader('Predicted class :')
  classes=["church","mosque","temple"]
  c2.write(classes[res_pred_classes[0]] )