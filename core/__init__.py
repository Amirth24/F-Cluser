import mtcnn
import streamlit as st

import numpy as np
import pandas as pd

from PIL import Image

from tensorflow.keras.models import load_model

from core.classifier import Classifier

__author__ = 'Amirth24'


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


    imgs_tab, tab2, summary = st.tabs(['Uploaded Images', 'Next Tab', 'Summary'])
    read_image_progress =tab2.progress(0)
    imgs_col = imgs_tab.columns(2)
    for i , file_ in enumerate(image_files):
        img = read_image(file_)
        imgs_col[i%2].image(img, use_column_width='auto')
        imgs.append(img)
        read_image_progress.progress(i/(n_images-1), 'Reading Files')

    read_image_progress.empty()
        
    faces  = []
    face_detector = get_mtcnn()
    detect_face_progress = tab2.progress(0)

    faces_data = pd.DataFrame(columns=['img_index', 'face_index'], )

    for i , img in enumerate(imgs):
        
        faces_detected = extract_faces(img, face_detector)

        for k in range(len(faces_detected)):
            print(faces_detected[k].max())
            faces_data = faces_data.append({
                'img_index': i, 'face_index' : k
            } , ignore_index=True)

        faces.extend(faces_detected)


        detect_face_progress.progress(i/(n_images-1),'Extracting Faces from the images')

    detect_face_progress.empty()
    faces_tensor = np.vstack(faces)



    # Load the model
    model = get_embed_model()

    face_embeds = model.predict(faces_tensor)


    classifier = Classifier()

    classifier.fit(face_embeds)


    face_labels = classifier.cluster_labels_

    n_faces = classifier.n_clusters

    faces_data['labels'] = classifier.predict(face_embeds)
    tab2.table(faces_data)


    write_summary(summary, n_images, face_embeds.shape[0], n_faces)
    

def write_summary(tab,n_images, n_f_samples, n_faces):
    summary = pd.DataFrame({
        'Values' :[n_images, n_f_samples, n_faces]
    }
    ,index= ['No. of Images', 'No. of Face Samples', 'No. of Faces Found']
    )


    tab.table(summary)
    