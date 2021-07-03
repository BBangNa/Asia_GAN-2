# import pandas as pd
# from sklearn.metrics.pairwise import linear_kernel
# from scipy.io import mmwrite, mmread
# import pickle
#
# df_review_one_sentence = pd.read_csv('./crawling/one_sentence_review_2016_2021.csv', index_col=0) # 데이터 프레임
# Tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()  # 매트릭스 불러옴
# with open('./models/tfidf.pickle', 'rb') as f:  # Tfidf도 불러옴
#     Tfidf = pickle.load(f)
#
# def getRecommendation(cosine_sin):   # 10개의 유사 데이터를 추천하는 함수
#     simScore = list(enumerate(cosine_sin[-1]))
#     simScore = sorted(simScore, key=lambda x:x[1], reverse=True)
#     simScore = simScore[1:10]
#     movieidx = [i[0] for i in simScore]
#     recMovieList = df_review_one_sentence.iloc[movieidx]
#     return recMovieList
#
# movie_idx = df_review_one_sentence[df_review_one_sentence['titles']=='말레피센트 2 (Maleficent: Mistress of Evil)'].index[0]
# # movie_idx = 127  #번 영화
# # print(df_review_one_sentence.iloc[movie_idx,0])
# cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation)
# 문장 유사도

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmwrite, mmread
import pickle
from gensim.models import Word2Vec

df_review_one_sentence = pd.read_csv('./crawling/one_sentence_review_2016_2021.csv',index_col=0)

"""
ls = ['겨울왕국', '라이온킹', '알라딘']
print(list(enumerate(ls))) # 튜플들의 리스트를 만들어 줌 인덱스 번호 포함.
for idx, i in enumerate(ls):
    if i == '라이온킹':
        print(idx)
    #print(idx, i)
"""

# 리스트 = [아이 + '짱 재밌당' for 아이 in ls]
# print(리스트)

Tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle','rb') as f :
    Tfidf = pickle.load(f)

def getRecommendation(cosine_sim) : # 벡터 공간에서 코사인 유사도 보는 것
    simScore = list(enumerate(cosine_sim[-1]))  # index를 붙이는 enumerate 사용함, -1값 : Tfidf_matrix
    simScore = sorted(simScore, key=lambda  x:x[1], reverse=True) # 데이터 유사도를 내림차순으로 정렬.
    simScore = simScore[1:11]  # 0번은 자기 자신이기 때문에 0을 빼준다. 1번부터 10번까지 뽑아낸다.
    movieidx = [i[0] for i in simScore] # 리스트를 for문을 통해 만듦.
    recMovieList = df_review_one_sentence.iloc[movieidx]  # 10개 목록을 리스트로 만들어 줌
    return  recMovieList

"""
print(df_review_one_sentence.iloc[0])
print(df_review_one_sentence.iloc[0,0]) # 0번째 행의 0번째 컬럼
print(df_review_one_sentence.loc[:,'reviews']) # --> DataFrame에 인덱싱하는 방법
print(df_review_one_sentence['titles'][0]) # 앞에 column명, 뒤에 인덱스 번호
print(df.iloc[row,col])
print(df.loc['tom', 'math'])
exit()
"""

# 제목을 알고 있을 때
# movie_idx = df_review_one_sentence[df_review_one_sentence['titles']=='라이온 킹 (The Lion King)'].index[0]

# 번호를 알고 있을 때 / 라이온킹 2185번, 그것 2080번
# movie_idx = 2185
# print(df_review_one_sentence.iloc[movie_idx,0])
#
# cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix)
# recommendation = getRecommendation(cosine_sim)
# print(recommendation.iloc[:,0])  # 제목만 보려면 recommendation.iloc[:,0]으로 출력하기, 0번 컬럼이 title.

# 키워드 기반으로 추천
embedding_model = Word2Vec.load('./models/word2VecModel_2016_2021.model')
key_word = '토르'
sentence = [key_word] * 10 # --> ['겨울', '겨울', '겨울', '겨울'...'겨울'] 나옴

sim_word = embedding_model.wv.most_similar(key_word, topn=10)
labels = []
for label, _ in sim_word:
    labels.append(label)
print(labels)
for i, word in enumerate(labels):
    sentence += [word] * (9-i)
sentence = ' '.join(sentence)
print(sentence)

sentence_vec = Tfidf.transform([sentence])
cosine_sim = linear_kernel(sentence_vec, Tfidf_matrix)
recommendation = getRecommendation(cosine_sim)
print(recommendation['titles'])