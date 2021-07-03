import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import collections
from konlpy.tag import Okt
import matplotlib as mpl
from matplotlib import font_manager, rc

""" 리눅스, 코랩인 경우 아래 코드를 이용해서 사용함 """
# import matplotlib as mpl
# import matplotlib.font_manager as fm
# fontpath = '/usr/share~~ 폰트경로.ttf'
# font = fm.FontProperties(fname=fontpath, size=9)
# plt.rc('font', family='NanumBarunGothic')
# mpl.font_manager._rebuild()


fontpath = './Jalnan.ttf'
font_name = font_manager.FontProperties(fname=fontpath).get_name()
rc('font', family=font_name)
mpl.font_manager._rebuild() # 한글제목

df = pd.read_csv('./crawling/one_sentence_review_2016_2021.csv', index_col=0)
df.dropna(inplace=True)
print(df.info())

print(df.head())

movie_index = df[df['titles'] == '슈퍼 햄찌 (SUPER FURBALL)'].index[0]
# print(movie_index)
print(df.reviews[movie_index])  # df.cleaned_senetences[movie_index]
words = df.reviews[movie_index].split(' ') # 띄어쓰기 기준으로 단어를 잘라서 문자열로 반환함 / df.cleaned_senetences[movie_index]
print(words)

worddict = collections.Counter(words) # 나온 단어의 빈도수를 나타냄
worddict = dict(worddict)
print(worddict)

stopwords = ['관객', '작품', '영화', '감독', '주인공', '출연', '개봉', '촬영']

wordcloud_img = WordCloud(background_color='white', max_words=2000, font_path=fontpath, stopwords=stopwords).generate(df.reviews[movie_index])
# wordcloud_img = WordCloud(background_color='white', max_words=2000, font_path=fontpath).generate_from_frequencies(worddict)

plt.figure(figsize=(8,8))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off')
plt.title(df.titles[movie_index], size=25)
plt.show()