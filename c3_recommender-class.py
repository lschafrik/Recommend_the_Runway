from surprise import SVD, accuracy
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise.reader import Reader
from surprise.dump import load

import numpy as np
import pandas as pd


surprise_model = load("data/svd_garment_model.pickle")
ratings_df = pd.read_csv('data/ratings_df.csv')
item_cat_df = pd.read_csv('data/item_category.csv')

# items_matrix = pd.read_csv('data/item_cos_matrix.csv')


class garmentRecommender():
    '''
    Rent the Runway garment recommender
    '''

    def __init__(self, model=surprise_model, ratings_matrix=ratings_df, category_data=item_cat_df):
        self.model = model
        self.algo = self.model[1]
        self.ratings_matrix = ratings_matrix
        self.category_data = category_data

    def predict(self, usr_id, n=5):
        '''
        Takes a User ID and creates a Pandas DataFrame of given user's rating predictions for all items

        INPUT:
            usr_id: INTEGER - User ID to perform rating predictions on
            n: INTEGER - Number of rows to return
        
        OUTPUT:
            Returns a Pandas DataFrame of 'n' rows of the highest rating predictions, sorted by descending values

        Notes: Column labels : Column Description = 'uid':User ID, 'iid':Item ID, and 'est':Predicted Rating
        '''

        item_lst = self.ratings_matrix['item_id'].unique()
        
        predictions = [self.algo.predict(usr_id, itm) for itm in item_lst]
        
        return pd.DataFrame(predictions)[['uid', 'iid', 'est']].sort_values(by='est',ascending=False)[0:n]


    def recommend_item(self, usr_id, n=5):
        '''
        Returns the top 'n' recommended items based on ratings_matrix and the item's related  category

        OUTPUT:
            self.recommended_item: LIST OF LISTS - [[INT(item_id), STR(category)], ...]

        Notes: Further descriptive information on items is currently unavailable; only recognizable by category
        '''

        recommended_items = []
        user_items = self.ratings_matrix[self.ratings_matrix['user_id'] == usr_id]['item_id']
        
        for itm in self.predict(usr_id, n)['iid']:
            if itm not in user_items:
                category_name = self.category_data[self.category_data['item_id'] == itm]['category'].to_list()[0]
                recommended_items.append([itm, category_name])
                
        return recommended_items



# if __name__ == "__main__":

    recommender = garmentRecommender()
#     recommender.predict(47002)

#     print(recommender.recommend_item())
