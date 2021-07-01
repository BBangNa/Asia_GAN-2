# -*- coding: utf-8 -*-
"""preprocess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qkhy4-DaRoyfa3DFQ3ShA2vCWwPua7l4
"""


import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./reviews_2018.csv', index_col=0)
print(df.head())

# a = '1a25ds94kl가나다라마'
# s = re.sub('[^가-힣]','',a) # 가부터 힣까지만 뽑아낸다.
# print(s)

# print(df.iloc[0,1]) # 0행 1열 뽑아오기
# print('===============================================================')
# sentence = re.sub('[^가-힣| ' ']','',df.iloc[0,1])
# print(sentence)

okt = Okt()

"""
token = okt.pos(sentence, stem=True) # -이었다 -> -이다 처럼 동사원형으로 변형해준다.
print(token) 
# 명사 동사 부사만 살릴 것!

df_token = pd.DataFrame(token, columns=['word', 'class'])
print(df_token.head())

# noun, verb, adj
df_cleaned_token = df_token[(df_token['class'] == 'Noun')]
print(df_cleaned_token.head())

df_cleaned_token = df_token[(df_token['class'] == 'Noun') |
                            (df_token['class'] == 'Verb') |
                            (df_token['class'] == 'Adjective')]
print(df_cleaned_token.head(20))
"""

# 불용어 제거하기
stopwords = pd.read_csv('./stopwords.csv', index_col=0)
print(stopwords.head())

movie_stopwords = ['영화', '배우', '감독']
stopwords_list = list(stopwords.stopword) + movie_stopwords

# words = []
# for word in df_cleaned_token['word']:
#   if len(word) > 1:
#     if word not in stopwords_list:
#       words.append(word)
# print(words)

# # 문장합치기
# cleaned_sentence = ' '.join(words)
# print(cleaned_sentence)

count = 0
cleaned_sentences = []
for sentence in df.reviews:
  count += 1
  if count % 10 == 0:
    print('.', end='')
  if count % 100 == 0:
    print('')
  sentence = re.sub('[^가-힣 | ' ']', '', sentence)
  token = okt.pos(sentence, stem=True) # 동사와 형용사는 원형으로 , 하나의 문장이 길수록 형태소 분류가 오래걸림
  df_token = pd.DataFrame(token, columns=['word', 'class'])
  df_cleaned_token = df_token[(df_token['class'] == 'Noun') | # 명사
                            (df_token['class'] == 'Verb') |   # 동사
                            (df_token['class'] == 'Adjective')]  # 형용사 추출
  words = []
  for word in df_cleaned_token['word']:
    if len(word) > 1:
      if word not in stopwords_list:
        words.append(word)
  cleaned_sentence = ' '.join(words)
  cleaned_sentences.append(cleaned_sentence)
df['cleaned_sentences'] = cleaned_sentences
print(df.head())

print(df.info())

df = df[['titles', 'cleaned_sentences']]
print(df.info())
df.to_csv('./cleaned_review_2018.csv')




