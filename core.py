import mtcnn
import streamlit as st

import numpy as np

from PIL import Image

@st.cache_resource
def get_mtcnn():
    return mtcnn.MTCNN()


def process_image(image_files):
    n_images = len(image_files)

    imgs = []

    read_image_progress = st.progress(0)
    detect_face_progress = st.progress(0)


    for i , file_ in enumerate(image_files):
        img = Image.open(file_)
        img = np.array(img)

        imgs.append(img)

        read_image_progress.progress(i/(n_images-1), 'Reading Files')
        
    faces  = []
    face_detector = get_mtcnn()
    for i , img in enumerate(imgs):
        face_detected = face_detector.detect_faces(img)

        
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

        detect_face_progress.progress(i/(n_images-1),'Extracting Faces from the images')


    face_tensors = np.vstack(faces)


    st.write('The Shape of face tensors', face_tensors.shape)
    st.write('Face Samples detected', face_tensors.shape[0])