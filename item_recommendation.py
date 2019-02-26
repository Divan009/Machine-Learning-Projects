
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

book = pd.read_csv('BX-Books.csv', sep=';',error_bad_lines=False, encoding="latin-1")
book.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'Publisher',
       'ImageURLS', 'ImageURLM', 'ImageURLL']
user = pd.read_csv('BX-Users.csv', sep=';',error_bad_lines=False, encoding="latin-1")
user.columns = ['userID','Location', 'Age']
rating = pd.read_csv('BX-Book-Ratings.csv', sep=';',error_bad_lines=False, encoding="latin-1")
rating.columns = ['userID', 'ISBN', 'bookRating']

# understanding my data

rating.head()
user.head()
book.head()

combine_book_rating = pd.merge(rating, book, on='ISBN')
combine_book_rating.columns
columns = ['yearOfPublication', 'Publisher','bookAuthor','ImageURLS', 'ImageURLM',
       'ImageURLL']
combine_book_rating = combine_book_rating.drop(columns, axis = 1)
combine_book_rating.head()

#grouping book by titles and a new column for total rating count

combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['bookTitle'])

bookRatingCount = (combine_book_rating.
                   groupby(by=['bookTitle'])['bookRating'].
                   count().
                   reset_index().
                   rename(columns={'bookRating':'totalRatingCount'})
                   [['bookTitle','totalRatingCount']])

bookRatingCount.head()

#combine the rating data with the total rating count data

R_totalRatingCount = combine_book_rating.merge(bookRatingCount, left_on = 'bookTitle', 
                                               right_on = 'bookTitle', how = 'inner')
R_totalRatingCount.head()

#stats of totalRatingCount 
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(bookRatingCount['totalRatingCount'].describe())

#about 1% of the books have received 50 or more ratings
print(bookRatingCount['totalRatingCount'].quantile(np.arange(.9,1,.01)))

popularity_threshold = 50
ratingPopularBk = R_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
ratingPopularBk.head()

#Filter data to users in India

combined = ratingPopularBk.merge(user, left_on = 'userID', right_on = 'userID', how = 'left')

indRating = combined[combined['Location'].str.contains("india")]
indRating = indRating.drop('Age', axis = 1)
indRating.head()

if not indRating[indRating.duplicated(['userID', 'bookTitle'])].empty:
    initial_rows = indRating.shape[0]
    
    print ('Initial dataframe shape {0}'.format(indRating.shape))
    indRating = indRating.drop_duplicates(['userID', 'bookTitle'])
    current_rows = indRating.shape[0]
    print ('New dataframe shape {0}'.format(indRating.shape))
    print ('Removed {0} rows'.format(initial_rows - current_rows))

indRating_pivot = indRating.pivot(index = "bookTitle", columns="userID", values = "bookRating").fillna(0)
indRating_matrix = csr_matrix(indRating_pivot.values)

#USing KNN
model_knn = NearestNeighbors(metric = 'cosine', algorithm='brute')
model_knn.fit(indRating_matrix)

query_index = np.random.choice(indRating_pivot.shape[0])
distances, indices = model_knn.kneighbors(indRating_pivot.iloc[query_index, :].values.reshape(1,-1), n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendation for {0}:\n'.format(indRating_pivot.index[query_index]))
    else:
        print("{0}: {1}, with distance of {2}:".format(i, indRating_pivot.index[indices.flatten()[i]],distances.flatten()[i]))


#MAtrix Factorization

indRating_pivot2 = indRating.pivot(index = 'userID', columns = 'bookTitle', values = 'bookRating').fillna(0)
indRating_pivot2.head()

indRating_pivot2.shape
X = indRating_pivot2.values.T
X.shape

SVD = TruncatedSVD(n_components=12, random_state=17)
matrix = SVD.fit_transform(X)
matrix.shape

#calcluate pearsons R corr() coeff for every book pair
corr = np.corrcoef(matrix)
corr.shape

indBookTitle = indRating_pivot2.columns
indBookTitle = list(indBookTitle)
ADangerousFortune = indBookTitle.index("Tell Me Your Dreams")
print(ADangerousFortune)

corr_dangFortune = corr[ADangerousFortune]






































