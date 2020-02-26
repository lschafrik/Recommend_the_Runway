# if user previous rating count >= 15:

    # take top 'n' rated items
        # find top 'n' most similar not-yet-rated-by-user items from item_matrix
            # predict rating on these items based on user's previous ratings
                # if rating >= 9, recommend item

from surprise import SVD, accuracy
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise.reader import Reader

import numpy as np
import pandas as pd

class garmentRecommender():

    def __init__(self):
        self.model = SVD(lr_all=0.005, n_epochs=20, n_factors=25, reg_all=0.1)
        self.ratings_matrix = None
        self.item_matrix = None
        self.usr_id = None
        self.item_lst = None

    def fit(self, ratings_matrix, r_scale):
        self.ratings_matrix = ratings_matrix
        self.r_scale = r_scale

        reader = Reader(rating_scale=r_scale)
        data = Dataset.load_from_df(self.ratings_matrix, reader)
        trainset = data.build_full_trainset()

        self.model.fit(trainset)

    def get_items(self):
        return np.unique(np.array(self.ratings_matrix['item_id']))

    def predict(self, usr_id):
        self.usr_id = usr_id
        self.item_lst = get_items()
        self.predictions = []

        for itm in self.item_lst:
            self.predictions.append(self.model.predict(usr_id, itm))

    