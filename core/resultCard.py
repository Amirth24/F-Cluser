

class ResultCard:
    df = None
    faces = None
    imgs = None
    def __init__(self, face_label, tab):
        self.face_label = face_label
        self.face_thumbnail = 'assets/imgs/avatar.png'
        self.face_container, self.result_container = tab.columns([1,5])

    @staticmethod
    def set_data(faces_df):
        ResultCard.df = faces_df

    @staticmethod
    def set_faces(faces):
        ResultCard.faces = faces

    @staticmethod
    def set_imgs(imgs):
        ResultCard.imgs = imgs
        
    def get_faces_locs(self):
        return ResultCard.df.loc[ResultCard.df['labels'] == self.face_label]
        
    def set_face_thumbnail(self, idx):
        self.face_thumbnail = ResultCard.faces[idx]

    def show(self):
        faces_locs = self.get_faces_locs()
        self.set_face_thumbnail(faces_locs.iloc[0,0])
        self.face_container.image(self.face_thumbnail,use_column_width='always')
        self.result_container.subheader('Found in')

        img_cols = self.result_container.columns(min(4, len(faces_locs)) )
        for  im_col, i in zip(img_cols, faces_locs['img_index']):
            im_col.image(ResultCard.imgs[i], use_column_width='always')
