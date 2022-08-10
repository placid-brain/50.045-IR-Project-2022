

import pandas as pd
import math
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ***************************************************************************************************************************************
# Query
#query = "aubergine fennel bulb red onion mushrooms olive oil garlic cloves"
#query = query.lower()
#q_list = query.split()
# print (q_list)

# ***************************************************************************************************************************************
# Read file
df = pd.read_csv("/Users/placid_brain/Documents/IR stuff/lsa/recipes_w_search_terms.csv",error_bad_lines=False, engine="python", nrows=1000)

# ***************************************************************************************************************************************
# Processing
def function(ini_list):
  new_cell = ini_list.strip('][').split(', ')
  for item in new_cell:
    item = item.replace("'","")
  return new_cell

df['ingredients'] = df['ingredients'].apply(function)
df['steps'] = df['steps'].apply(function)
df1 = df[['id', 'name','ingredients','steps']]
df1.to_pickle("./df1_926.pkl")  
# print (df1)
# print(df1['ingredients'])
# print ((df1['ingredients'][0]))
# print ((df1['ingredients'][1]))




# ***************************************************************************************************************************************
# tf-idf vector for documents
data = []
for i in range (len(df1['ingredients'])):
    ingre_str = ''.join(df1['ingredients'][i])
    # print (ingre_str)
    data.append(ingre_str)

# print(data)

tfvec = TfidfVectorizer()
tdf = tfvec.fit_transform(data)
tf_idf_docs_matrix = pd.DataFrame(tdf.toarray(), columns = tfvec.get_feature_names())

'''term_list = tfvec.get_feature_names()
with open('list_pickle', 'wb') as fh:
  pickle.dump(term_list, fh)
tf_idf_docs_matrix.to_pickle("./tf_idf_docs_matrix.pkl") '''
# print(tf_idf_docs_matrix)

# get list of tf-idf vectors of recipes
# print (tf_idf_docs_matrix.values.tolist())

# ***************************************************************************************************************************************

def vector_gen(q_list):
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

# print (query_tfidf_vector)
# print(len(query_tfidf_vector))


def search(query):
  sw = set(stopwords.words('english'))
  
  
  query = query.lower()
  
  

  
  query_tokens = word_tokenize(query)
  q_list = []

  for w in query_tokens:
      if w not in sw:
          q_list.append(w)
  query_tfidf_vector = vector_gen(q_list)

  
  # similarity - cosine similarity
  # dict to store cosine similarity between query and documents (key - doc index, value - score)
  similarity_list = []

  for i in range(len(tf_idf_docs_matrix.values.tolist())):
      # print(tf_idf_docs_matrix.values.tolist()[i])
      cosine_similarity = 1 - spatial.distance.cosine(query_tfidf_vector, tf_idf_docs_matrix.values.tolist()[i])
      similarity_list.append(cosine_similarity)

  # print(similarity_list)

  top_5_recipes = sorted(range(len(similarity_list)), key=lambda i: similarity_list[i], reverse=True)[:5]

  # print(top_5_recipes)

  for i in range(len(top_5_recipes)):
    # print(top_5_recipes[i])
    print("{}. {}".format(i+1, df1['name'][top_5_recipes[i]]))



#search('aubergine fennel bulb red onion mushrooms olive oil garlic cloves')

#pickle.dump(model, open("model", 'wb'))


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
      
      # similarity - cosine similarity
      # dict to store cosine similarity between query and documents (key - doc index, value - score)
      similarity_list = []
      final_list = []

      for i in range(len(tf_idf_docs_matrix.values.tolist())):
          # print(tf_idf_docs_matrix.values.tolist()[i])
          cosine_similarity = 1 - spatial.distance.cosine(query_tfidf_vector, tf_idf_docs_matrix.values.tolist()[i])
          similarity_list.append(cosine_similarity)

      # print(similarity_list)

      top_5_recipes = sorted(range(len(similarity_list)), key=lambda i: similarity_list[i], reverse=True)[:5]

      for i in range(len(top_5_recipes)):
        final_list.append(df1['name'][top_5_recipes[i]])
        print("{}. {}".format(top_5_recipes[i], df1['name'][top_5_recipes[i]]))
      return final_list
    else:
      ["No matching results found"]
     
    



vsm = VSM()
vsm.search('aubergine fennel bulb red onion mushrooms olive oil garlic cloves')

with open('model_pickle_926','wb') as f:
  pickle.dump(vsm,f)











