import pandas as pd

df = pd.read_csv('crawling/cleaned_review_2018.csv', index_col=0)
df.dropna(inplace=True)
df.to_csv('./cleaned_review_2018.csv')
# print(df.head())
# print(df.info())

# temp = df[df['titles']==title]['cleaned_sentences']
# one_sentence = ' '.join(temp)
# one_sentences.append(one_sentence)

one_sentences = []
for idx, title in enumerate(df['titles'].unique()):
    print(idx)
    print(title)
    temp = df[df['titles']==title]['cleaned_sentences']
    one_sentence = ' '.join(temp)
    one_sentences.append(one_sentence)  # Nan값이 있어서 오류가 났음
df_one_sentences = pd.DataFrame({'titles':df['titles'].unique(), 'reviews':one_sentences})
print(df_one_sentences.head())
df_one_sentences.to_csv('./one_sentence_review_2018.csv')














