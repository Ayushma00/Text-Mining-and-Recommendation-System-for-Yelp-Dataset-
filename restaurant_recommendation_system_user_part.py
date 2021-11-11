import numpy as np
import pandas as pd
import warnings
import pickle as pkl
import string
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')

def main():
    with open("./recommendation system files/yelp_recommendation_model_Q.pkl", "rb") as f:
        Q = pkl.load(f)

    with open("./recommendation system files/yelp_recommendation_model_userid_vectorizer.pkl", "rb") as f:
        userid_vectorizer = pkl.load(f)

    restaurant = pd.read_csv("./recommendation system files/restaurant.csv", index_col=0)
    restaurant_df= pd.read_csv("./recommendation system files/restaurant_cluster_df.csv", index_col=0)
    cosine_similarity_df = pd.read_csv("./recommendation system files/cosine_similarity_restaurant_df.csv", index_col=0)

    stop = []
    for word in stopwords.words('english'):
        s = [char for char in word if char not in string.punctuation]
        stop.append(''.join(s))

    def text_process(mess):
        nopunc = [char for char in mess if char not in string.punctuation]
        nopunc = ''.join(nopunc)
        return " ".join([word for word in nopunc.split() if word.lower() not in stop])

    def recommender(name):
        try:
            given_restaurant=restaurant_df[restaurant_df["name"]==name]
            same_cluster = restaurant_df[restaurant_df["cluster"] == list(given_restaurant.cluster.items())[0][1]]
            distance = np.sqrt((same_cluster.latitude - list(given_restaurant.cluster.items())[0][1])**2 + (same_cluster.longitude - list(given_restaurant.cluster.items())[0][1])**2)
            same_cluster["distance"] = distance
            same_cluster.sort_values('distance',inplace=True)
            recommended_restaurant = same_cluster.iloc[1:6]
            for i in recommended_restaurant["business_id"]:
                print(restaurant[restaurant['business_id']==i]['name'].iloc[0])
                print(restaurant[restaurant['business_id']==i]['categories'].iloc[0])
                print(str(restaurant[restaurant['business_id']==i]['stars'].iloc[0])+ ' '+str(restaurant[restaurant['business_id']==i]['review_count'].iloc[0]))
                print('')
            try:
                for i in cosine_similarity_df.iloc[cosine_similarity_df[name].sort_values(ascending=False).index].name[1:6].items():
                    print(restaurant[restaurant['name']==i[1]]['name'].iloc[0])
                    print(restaurant[restaurant['name']==i[1]]['categories'].iloc[0])
                    print(str(restaurant[restaurant['name']==i[1]]['stars'].iloc[0])+ ' '+str(restaurant[restaurant['name']==i[1]]['review_count'].iloc[0]))
                    print('')
            except:
                pass
        except:
            print("Restaurant Doesn't Exists!!\n")

    def text_Recommender(text):
        test_df= pd.DataFrame([text], columns=['text'])
        test_df['text'] = test_df['text'].apply(text_process)
        test_vectors = userid_vectorizer.transform(test_df['text'])
        test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())

        predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
        topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:5]

        for i in topRecommendations.index:
            print(restaurant[restaurant['business_id']==i]['name'].iloc[0])
            print(restaurant[restaurant['business_id']==i]['categories'].iloc[0])
            print(str(restaurant[restaurant['business_id']==i]['stars'].iloc[0])+ ' '+str(restaurant[restaurant['business_id']==i]['review_count'].iloc[0]))
            print('')

    while True:
        ans = int(input("Do you want to input restaurant name or query.\nType 0 for restaurant name.\nType 1 for query.\n"))
        if ans==0:
            restaurant_name = input("Enter name of the restaurant: ")
            recommender(restaurant_name)
        elif ans==1:
            text_query = input("Enter your query: ")
            text_Recommender(text_query)
        else:
            print("Invalid Input!!")
        redo = input("Do you want to run again? [y/n]")
        if redo=="n":
            break

if __name__=="__main__":
    main()
