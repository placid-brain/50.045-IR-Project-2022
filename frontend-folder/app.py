
import streamlit as st
import pickle
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


df1 = pd.read_pickle("./df1_109.pkl")
data = []
for i in range (len(df1['ingredients'])):
    ingre_str = ''.join(df1['ingredients'][i])
    # print (ingre_str)
    data.append(ingre_str)

tfvec = TfidfVectorizer()
tdf = tfvec.fit_transform(data)
tf_idf_docs_matrix = pd.DataFrame(tdf.toarray(), columns = tfvec.get_feature_names())






class VSM:
  def __init__(self):
    #super(VSM, self).__init__()
    #self.query = query
    #self.search(self.query)
    #super(VSM, self).__init__()
    pass
  def vector_gen(self, q_list):
    # tf-idf vector for query
    # print(tfvec.get_feature_names())

    query_tfidf_vector = []
    for i in tfvec.get_feature_names():
        # print(i)
        if i not in q_list:
            query_tfidf_vector.append(0.0)
        else:
            # append tf-idf of the term for query
            # print(i)
            query_tfidf_vector.append(tfvec.idf_[tfvec.vocabulary_[i]])
    return query_tfidf_vector

  def search(self,query):
    query = query.lower()
    q_list = query.split()
    if set(q_list).issubset(set(tfvec.get_feature_names()))== True:
        query_tfidf_vector = self.vector_gen(q_list)
        '''query_tfidf_vector = []
        
        for i in tfvec.get_feature_names():
            # print(i)
            if i not in q_list:
                query_tfidf_vector.append(0.0)
            else:
                # append tf-idf of the term for query
                # print(i)
                query_tfidf_vector.append(tfvec.idf_[tfvec.vocabulary_[i]])'''
        
        # similarity - cosine similarity
        # dict to store cosine similarity between query and documents (key - doc index, value - score)
        similarity_list = []
        final_list=[]
        for i in range(len(tf_idf_docs_matrix.values.tolist())):
            # print(tf_idf_docs_matrix.values.tolist()[i])
            cosine_similarity = 1 - spatial.distance.cosine(query_tfidf_vector, tf_idf_docs_matrix.values.tolist()[i])
            similarity_list.append(cosine_similarity)

        # print(similarity_list)

        top_5_recipes = sorted(range(len(similarity_list)), key=lambda i: similarity_list[i], reverse=True)[:5]

        # print(top_5_recipes)

        for i in range(len(top_5_recipes)):
        # print(top_5_recipes[i])
            final_list.append(df1['name'][top_5_recipes[i]])
            print("{}. {}".format(i+1, df1['name'][top_5_recipes[i]]))
        return final_list
    else:
        return ["No matching results found"]






with open('model_pickle_109','rb') as f:
    

    mp = pickle.load(f)


mp.search("chicken honey")

st.title("Food Dishes Search Engine")

x = st.text_area(label="Type in your available ingredients")

btn = st.button('Get dishes')

if btn:
    result = mp.search(x)
    if result[0]=='No matching results found':
        st.subheader('No matching results found')
    else:
        for idx in range(len(result)):
            st.subheader("{}. {}".format(idx+1, result[idx]))
    