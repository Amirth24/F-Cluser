from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class Classifier:
    def __init__(self, with_pca=False) -> None:
        self.__with_pca = with_pca
        self.__best_mdl, self.__best_score_, self.__best_k = (None, 0, 2)
        self.__cluster_labels_ = []

    @property
    def cluster_labels_(self):
        if self.__cluster_labels_ == []:
            raise Exception("ModelNotFitException")

        return self.__cluster_labels_

    @property
    def n_clusters(self):
        return self.__best_k

    @property
    def best_score(self):
        return self.__best_score_

    def fit(self, fe_vecs):
        if self.__with_pca:
            self.__fit_with_pca(fe_vecs)
        else:
            self.__fit(fe_vecs)

    def __fit(self, fe_vecs):

                
        best_mdl, best_score, best_k = (None, 0, 2)
        ks = range(2,int(fe_vecs.shape[0]))
        cluster_labels= None
        for k in ks:
            mdl = GaussianMixture(n_components=k).fit(fe_vecs)
            cluster_labels = mdl.predict(fe_vecs)
            st_score = silhouette_score(fe_vecs, cluster_labels)
            
            if best_score < st_score: best_mdl, best_score , best_k = mdl, st_score, k    

        self.__best_mdl = best_mdl
        self.__best_k = best_k
        self.__cluster_labels_ = cluster_labels

    def predict(self, fe_vec):
        if self.__best_mdl is None:
            raise Exception("ModelNotFitException")
        if self.__with_pca:
            transformed_vec = self.pca_transformer.transform(fe_vec)

            return self.__best_mdl.predict(transformed_vec)

        else:
            return self.__best_mdl.predict(fe_vec)

    def __fit_with_pca(self, fe_vecs):
        self.pca_transformer = PCA(n_components=min(64, fe_vecs.shape[0]))

        transformed_vecs = self.pca_transformer.fit_transform(fe_vecs)

        self.__fit(transformed_vecs)
