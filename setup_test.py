from core import get_embed_model, read_image, extract_faces, get_mtcnn
import numpy as np


imgs = [read_image(f'test_imgs/img{i}.jpg') for i in range(2,6)]

mdl = get_mtcnn()

faces = []

for im in imgs:
    faces.extend(extract_faces(im, mdl))

f_tnsr = np.vstack(faces)

embeds = get_embed_model().predict(f_tnsr)


np.save('test_imgs/face_embeds.npy', embeds)