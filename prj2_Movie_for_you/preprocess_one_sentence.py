import pandas as pd

df = pd.read_csv('crawling/movie_review_2016_2021.csv', index_col=0)
df.dropna(inplace=True)
df.to_csv('./crawling/movie_review_2016_2021_A.csv')
# print(df.head())
# print(df.info())

# temp = df[df['titles']==title]['cleaned_sentences']
# one_sentence = ' '.join(temp)
# one_sentences.append(one_sentence)

one_sentences = []
for idx, title in enumerate(df['titles'].unique()):
    # print(idx)
    # print(title)
    temp = df[df['titles']==title]['cleaned_reviews']
    one_sentence = ' '.join(temp)
    one_sentences.append(one_sentence)  # Nan값이 있어서 오류가 났음
df_one_sentences = pd.DataFrame({'titles':df['titles'].unique(), 'reviews':one_sentences})
print(df_one_sentences.head())
df_one_sentences.to_csv('./crawling/one_sentence_review_2016_2021.csv')
print(df_one_sentences.info())














