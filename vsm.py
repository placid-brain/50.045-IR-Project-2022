import pandas as pd
import math
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

# ***************************************************************************************************************************************
# Query
query = "aubergine fennel bulb red onion mushrooms olive oil garlic cloves"
query = query.lower()
q_list = query.split()
# print (q_list)

# ***************************************************************************************************************************************
# Read file
df = pd.read_csv("recipes_w_search_terms.csv",error_bad_lines=False, engine="python", nrows=1000)

# ***************************************************************************************************************************************
# Processing
def function(ini_list):
  new_cell = ini_list.strip('][').split(', ')
  for item in new_cell:
    item = item.replace("'","")
  return new_cell

df['ingredients'] = df['ingredients'].apply(function)
df1 = df[['id', 'name','ingredients','steps']]
# print (df1)
# print(df1['ingredients'])
# print ((df1['ingredients'][0]))
# print ((df1['ingredients'][1]))


# prints out index then prints out list of ingredients
# for i in range (len(df1['ingredients'])):
#   print (i)
#   print (df1['ingredients'][i])

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
# print(tf_idf_docs_matrix)

# get list of tf-idf vectors of recipes
# print (tf_idf_docs_matrix.values.tolist())

# ***************************************************************************************************************************************
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

# print (query_tfidf_vector)
# print(len(query_tfidf_vector))


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
