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

# garments_df = 

ratings_df = pd.read_csv('data/ratings_df.csv')
# items_matrix = pd.read_csv('data/item_cos_matrix.csv')

class garmentRecommender():

    def __init__(self, model, ratings_matrix, items_matrix=None):
        self.model = model
        self.ratings_matrix = ratings_matrix
        self.items_matrix = items_matrix

    def translate(self, r_scale):
        self.r_scale = r_scale

        reader = Reader(rating_scale=r_scale)
        data = Dataset.load_from_df(self.ratings_matrix, reader)

        self.trainset = data.build_full_trainset()
        # return self.trainset

    def fit(self):
        return self.model.fit(self.trainset)

    def predict(self, usr_id, n=10):
        self.usr_id = usr_id
        self.item_lst = np.unique(np.array(self.ratings_matrix['item_id']))
        self.predictions = []

        for itm in self.item_lst:
            self.predictions.append(self.model.predict(usr_id, itm))
        
        self.predictions = pd.DataFrame(self.predictions)[['uid', 'iid', 'est']].sort_values(by='est',
                                                                                             ascending=False)[0:n]
        return self.predictions

    def recommend_item(self):
        self.recommended_item = []
        self.user_items = self.ratings_matrix[self.ratings_matrix['user_id'] == self.usr_id]['item_id']
        for itm in self.predictions['iid']:
            if itm not in self.user_items:
                self.recommended_item.append(itm)

        return self.recommended_item



if __name__ == "__main__":

    model = SVD(lr_all=0.005, n_epochs=20, n_factors=25, reg_all=0.1)
    recommender = garmentRecommender(model, ratings_df)

    # print(recommender.translate((2,10)))
    recommender.translate((2,10))

    # print(recommender.fit())
    recommender.fit()

    # print(recommender.predict(691468, 5))
    recommender.predict(691468, 5)

    print(recommender.recommend_item())
