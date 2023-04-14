import mtcnn
import streamlit as st

import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model

@st.cache_resource
def get_mtcnn():
    return mtcnn.MTCNN()

@st.cache_resource
def get_embed_model():
    return load_model('model/keras_facenet.h5')

@st.cache_resource(show_spinner=False)
def read_image(image):
    img = Image.open(image)
    return np.array(img)

@st.cache_data(show_spinner=False)
def extract_faces(img, _mdl):
    face_detected = _mdl.detect_faces(img)

    faces = []
    for face in face_detected:
        x1, y1, width, height = face['box']

        x1,y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        face_detected_img = img[y1:y2, x1:x2]
        
        pil_img = Image.fromarray(face_detected_img)
        resized_img = pil_img.resize((160, 160))

        face_img = np.asarray(resized_img)
        mean, std = face_img.mean(), face_img.std()
        face_img =( face_img - mean)/ std # z Scale
        face_img = np.expand_dims(face_img, axis=0)
        faces.append(face_img)
    return faces


@st.cache_data(show_spinner=False)
def process_image(image_files):
    n_images = len(image_files)

    imgs = []

    read_image_progress = st.progress(0)


    for i , file_ in enumerate(image_files):
        
        imgs.append(read_image(file_))
        read_image_progress.progress(i/(n_images-1), 'Reading Files')

    read_image_progress.empty()
        
    faces  = []
    face_detector = get_mtcnn()
    detect_face_progress = st.progress(0)

    for i , img in enumerate(imgs):
            
        faces.extend(extract_faces(img, face_detector))

        detect_face_progress.progress(i/(n_images-1),'Extracting Faces from the images')

    detect_face_progress.empty()
    face_tensors = np.vstack(faces)




    # Load the model
    model = get_embed_model()

    face_embeds = model.predict(face_tensors)

    