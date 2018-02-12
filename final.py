import pandas as pd
import gzip
import time
import numpy as np
import scipy.sparse as ss

'''
Code references:
http://snap.stanford.edu/data/amazon/productGraph/
https://docs.scipy.org/doc/scipy/reference/sparse.csgraph.html
https://stackoverflow.com/questions/38688062/converting-a-1-2gb-list-of-edges-into-a-sparse-matrix
'''

print("DATA LOADING... ")
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Electronics_5.json.gz')
print("DATA LOADED... ")
#print(df)

### NUMBER OF UNIQUE FOLLOWERS AND PRODUCTS ###
print("FINDING UNIQUE FOLLOWERS, PRODUCTS AND REVIEWS... ")
number_unique_reviewers = df.reviewerID.unique()
print("NUMBER OF UNIQUE REVIEWERS: {}".format(len(number_unique_reviewers)))

number_unique_products = df.asin.unique()
print("NUMBER OF UNIQUE PRODUCTS: {}".format(len(number_unique_products)))

number_unique_reviews = df.shape[0]
print("NUMBER OF UNIQUE REVIEWS: {}".format(number_unique_reviews))

### LARGEST DEGREES - POTENTIALLY MINIMUM NUMBER OF COLORS = MAX NEIGHBORS
### GRAPH IS NOT COMPLETE OR BIPARTITE ###

print("MODES OF REVIEWERS - TOP THREE :\n{}".format(df["reviewerID"].value_counts()[:3]))
print("MODES OF PRODUCTS - TOP THREE :\n{}".format(df["asin"].value_counts()[:3]))

### NUMBER OF CONNECTED COMPONENTS ###
print("FINDING CONNECTED COMPONENTS... ")
subsetdf = df[["reviewerID", "asin"]]
#print(subsetdf)
graphdf = subsetdf.stack().rank(method='dense').unstack()

rows = graphdf["reviewerID"]
cols = graphdf["asin"]

shape = max(tuple(graphdf.max(axis=0)[:"PRODUCT"]+1))
#print(shape)

ones = np.zeros(len(rows), np.uint32)
matrix = ss.coo_matrix((ones, (rows, cols)), shape = (shape,shape))
#print(matrix)



x,y = ss.csgraph.connected_components(matrix,directed=False,connection='strong',return_labels=True)
print("SCC: {}".format(x-1))

x,y = ss.csgraph.connected_components(matrix,directed=False,connection='weak',return_labels=True)
print("WCC-DISCONNECTED CLUSTERS: {}".format(x-1))

### RETURN REVIEW FROM PRODUCT ID AND ASSOCIATED REVIEWER ID ("asin") ###
print("EXAMPLE OF PRODUCT ID -> REVIEW FUNCTION... ")
def return_reviewText_from_ProductID(asin):
    ID = str(asin)
    return "\nProduct ID : {} \nReviews : \n{}\n".format( ID, df.loc[ (df['asin'] == ID) ,["reviewText", "reviewerID"]])

print(return_reviewText_from_ProductID("0528881469"))


### RETURN REVIEW OF PRODUCT ASSOCIATED WITH REVIEWER ID ("reviewerID") ###
print("EXAMPLE OF REVIEWER ID -> REVIEW FUNCTION... ")
def return_reviewText_from_ReviewerID(ID):
    ID = str(ID)
    return "\nReviewer ID : {} \nReviews : \n{}\n".format( ID, df.loc[ (df['reviewerID'] == ID) ,["reviewText", "asin"]])

print(return_reviewText_from_ReviewerID("AO94DHGC771SJ"))

#print(subsetdf) # unsorted

### RETURN DATA IN ASCENDING ORDER BY "asin" ###
print("SORTING DATA BY 'asin'... ")
subsetdf = subsetdf.sort_values("asin")
#print(subsetdf) # sorted
asin_sorted = subsetdf.index.tolist()
#print(df.ix[0])
print("{} position {}\n".format(df.ix[asin_sorted[100000]], 100000))
print("{} position {}\n".format(df.ix[asin_sorted[200000]], 200000))
print("{} position {}\n".format(df.ix[asin_sorted[300000]], 300000))
print("{} position {}\n".format(df.ix[asin_sorted[400000]], 400000))
print("{} position {}\n".format(df.ix[asin_sorted[500000]], 500000))
