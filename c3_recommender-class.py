from surprise import SVD, accuracy
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise.reader import Reader
from surprise.dump import load

import numpy as np
import pandas as pd
import time


surprise_model = load("data/svd_garment_model.pickle")
ratings_df = pd.read_csv('data/ratings_df.csv')
item_cat_df = pd.read_csv('data/item_category.csv').set_index('item_id')['category']

items_matrix = pd.read_csv('data/item_cos_matrix.csv')
items_matrix.set_index(items_matrix.columns, inplace=True)

nmf_matrix = pd.read_csv('data/nmf_matrix.csv')
nmf_matrix.set_index(ratings_df.user_id.sort_values().unique(), inplace=True)

class garmentRecommender:
    '''
    Rent the Runway garment recommender
    '''

    def __init__(self, model=surprise_model, ratings_matrix=ratings_df, category_data=item_cat_df, item_matrix=items_matrix, nmf_matrix=nmf_matrix):
        self.model = model
        self.algo = self.model[1]
        self.category_data = category_data
        self.ratings_matrix = ratings_matrix
        self.item_matrix = item_matrix
        self.nmf_matrix = nmf_matrix

    def __predict__(self, usr_id, n=5):
        '''
        Takes a User ID and creates a Pandas DataFrame of given user's rating predictions for all items

        INPUT:
            usr_id: INTEGER - User ID to perform rating predictions on
            n: INTEGER(Opt) - Number of rows to return
        
        OUTPUT:
            Returns a Pandas DataFrame of 'n' rows of the highest rating predictions, sorted by descending values

        Notes: Column labels : Column Description = 'uid':User ID, 'iid':Item ID, and 'est':Predicted Rating
        '''

        item_lst = self.ratings_matrix['item_id'].unique()
        
        predictions = [self.algo.predict(usr_id, itm) for itm in item_lst]

        
        return pd.DataFrame(predictions)[['uid', 'iid', 'est']].sort_values(by='est',ascending=False)[0:n]

    def recommend_svd(self, usr_id, n=5):
        '''
        Returns the top 'n' recommended items based on ratings_matrix and the item's related  category

        INPUT:
            usr_id: INTEGER - User ID to perform item-item recommendations on
            n: INTEGER(Opt) - Number of rows to return

        OUTPUT:
            self.recommended_item: LIST OF LISTS - [[INT(item_id), STR(category)], ...]

        Notes: Further descriptive information on items is currently unavailable; only recognizable by category
        '''

        recommended_items = dict()
        user_items = self.ratings_matrix[self.ratings_matrix['user_id'] == usr_id]['item_id']
        
        for itm in self.__predict__(usr_id, n)['iid']:
            if itm not in user_items:
                category_name = self.category_data[itm]
                recommended_items[itm] = category_name
                
        recommended_df =  pd.DataFrame.from_dict(recommended_items, orient='index', columns=['category'])
        recommended_df.index.name = 'item_id'

        return recommended_df

    def recommend_item(self, usr_id, n=5, similarity_limit=0.99):
        '''
        Takes a User ID and creates a Pandas DataFrame of item recommendations based 
        on user's highest-rated item_id's similarity with other items in item_matrix

        INPUT:
            usr_id: INTEGER - User ID to perform item-item recommendations on
            n: INTEGER(Opt) - Number of rows to return
            similarity_limit: FLOAT(Opt) - Similarity limitation to specify how similar recommendations should be
        
        OUTPUT:
            Returns a Pandas DataFrame of 'n' rows of recommendations based on highest similarity to user's highest-rated item, sorted by descending values

        Notes: Further descriptive information on items is currently unavailable; only recognizable by category
        '''

        user_items = self.ratings_matrix[self.ratings_matrix['user_id'] == usr_id]
        user_fave_itm = str(int(user_items.sort_values(by='rating', ascending=False).iloc[0].item_id))
        similar_items_id = items_matrix[items_matrix[user_fave_itm] < similarity_limit][user_fave_itm].sort_values(ascending=False)[0:n].index

        recommended_items = dict()

        for itm in similar_items_id:
            if int(itm) not in user_items:
                category_name = self.category_data[int(itm)]
                recommended_items[itm] = category_name
                
        recommended_df =  pd.DataFrame.from_dict(recommended_items, orient='index', columns=['category'])
        recommended_df.index.name = 'item_id'

        return recommended_df

    def recommend_nmf(self, usr_id, n=5):
            '''
            Takes a User ID and creates a Pandas DataFrame of item recommendations based 
            on user's highest-rated item_id's similarity with other items in item_matrix

            INPUT:
                usr_id: INTEGER - User ID to perform item-item recommendations on
                n: INTEGER(Opt) - Number of rows to return
            
            OUTPUT:
                Returns a Pandas DataFrame of 'n' rows of recommendations based on highest similarity to user's highest-rated item, sorted by descending values

            Notes: Further descriptive information on items is currently unavailable; only recognizable by category
            '''

            user_items = self.ratings_matrix[self.ratings_matrix['user_id'] == usr_id]
            similar_items_id = self.nmf_matrix.loc[usr_id].sort_values(ascending=False)[0:n].index


            recommended_items = dict()

            for itm in similar_items_id:
                if int(itm) not in user_items:
                    category_name = self.category_data[int(itm)]
                    recommended_items[itm] = category_name
                    
            recommended_df =  pd.DataFrame.from_dict(recommended_items, orient='index', columns=['category'])
            recommended_df.index.name = 'item_id'

            return recommended_df



def recommend_something(recommender_results, recommender_name, usr, choice_count=4):
    if recommender_name not in ['svd', 'item', 'nmf']:
        return "'recommender_name' entry invalid, please enter one of the following: 'svd', 'item', or 'nmf'.  Thank you!"
    else:
        prep_phrases = np.array(["Sifting through socks...", "Trying on trousers...", "Contemplating coats...", 
        "Meticulously prepping mannequins...", "Ruminating over rompers...", "Sorting through sweaters...", 
        "Tailoring turtlenecks...", "Judging jumpsuits...", "Pondering pants...", "Deliberating denim...", 
        "Speculating over suits...", "Viewing various vests..."])

        choice_phrases = np.random.choice(prep_phrases, choice_count, replace=False)

        print()

        print(f"Hello, User {usr}, and thank you for choosing Recommend the Runway!  \
Please hold while we find the perfect outfit for you!")

        print()
        time.sleep(3)

        for chce in choice_phrases:
            print(chce)
            time.sleep(1)

        print()

        if recommender_name == "svd":
            print(f"Wonderful news, User {usr}!  Based on previous ratings made by yourself and other similar users, \
we think you'll love Item {recommender_results.index[0]}.  You have great taste - it's a \
wonderful {recommender_results['category'].iloc[0] }!")
        elif recommender_name == "item":
            print(f"What fun, User {usr}!  Based on your highest rated item, we have found another similar item we \
think you'll greatly enjoy; your taste is exquisite - Item {recommender_results.index[0]} is a \
breathtaking {recommender_results['category'].iloc[0] }!")
        elif recommender_name == "nmf":
            print(f"Oh my, User {usr}!  Based on previously rated items, we have located an item we hope to dazzle \
you with: Item {recommender_results.index[0]} - what a {recommender_results['category'].iloc[0] }!")
        
        time.sleep(3)
        print()
        print(f"Farewell, User {usr} - we hope to see you again soon!")


# if __name__ == "__main__":

    # r = garmentRecommender()
    # usr = 47002
    # svd_r = r.recommend_svd(usr)
    # print(recommend_something(svd_r, 'svd', usr))