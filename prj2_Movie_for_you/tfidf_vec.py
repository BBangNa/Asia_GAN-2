# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from scipy.io import mmwrite, mmread
# import pickle
#
# df_review_one_sentences = pd.read_csv('./crawling/one_sentence_review_2016_2021.csv', index_col=0)
# print(df_review_one_sentences.info())  # 데이터 크기 확인, null값 확인
#
# Tfidf = TfidfVectorizer(sublinear_tf=True)
# Tfidf_matrix = Tfidf.fit_transform(df_review_one_sentences['reviews']) # 표로 반환받음, 유사도 점수 표
#
# with open('./models/tfidf.pickle', 'wb') as f:
#     pickle.dump(Tfidf, f) # f에 ftidf를 담근다
#
# mmwrite('./models/tfidf_movie_review.mtx', Tfidf_matrix) # matrix를 저장할 때 사용하는 모듈, 저장방식은 Tfidf_matrix로

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread #입출력 관련
import pickle

df_reveiw_one_senteneces = pd.read_csv('./crawling/one_sentence_review_2016_2021.csv',index_col=0)
print(df_reveiw_one_senteneces.info())

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_reveiw_one_senteneces['reviews'])

#피클로 저장
with open('./models/tfidf.pickle','wb') as f :
    pickle.dump(Tfidf, f)

mmwrite('./models/tfidf_movie_review.mtx', Tfidf_matrix)





























