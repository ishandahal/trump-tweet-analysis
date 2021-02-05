import plotly.express as px
import datetime
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import nltk
nltk.download('wordnet')
from wordcloud import STOPWORDS, WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("trump_insult_tweets_2014_to_2021.csv.zip")

print(df.tail())

print(df.shape)

print(df.isna().sum())

df = df.fillna("Unknown")

print(df.isna().sum() / len(df))

Trumptargets = pd.DataFrame(df.groupby("target")["target"].count()).rename(columns={"target":"#ofattacks"}).reset_index().sort_values(by="#ofattacks").reset_index(drop=True)
print(Trumptargets)

Trumptargets_ = Trumptargets[Trumptargets["#ofattacks"] > 50]
print(Trumptargets_)

fig = px.bar(Trumptargets_, x="#ofattacks", y="target", orientation="h", height=900,
            title="Most frequent targets")
fig.show()

df['year'] = pd.DatetimeIndex(df["date"]).year
df['month'] = pd.DatetimeIndex(df["date"]).month
df['day'] = pd.DatetimeIndex(df["date"]).day

df_2 = df[df["year"] > 2020]
print(df_2.shape)
print(df_2.tail())

print(f"Trump's last hateful message:\n{df_2.loc[10359][3]} \n\nWhich was aimed at \n{df_2.loc[10359][2]}")

Trumptargets__ = pd.DataFrame(df_2.groupby("target")["target"].count()).rename(columns =\
                {"target" : "#ofattacks"}).reset_index().sort_values(by="#ofattacks").reset_index(drop=True)
print(Trumptargets__)

fig = px.bar(Trumptargets__, x="#ofattacks", y="target", height=500,
            title="Tweets of Trump in the final days (2021)")
fig.show()

stopwords = set(STOPWORDS)

text1 = df_2["tweet"].to_csv()
tweet1_wc = WordCloud(
        background_color='white',
        max_words=1000,
        stopwords=stopwords)

tweet1_wc.generate(text1)

plt.figure(figsize=(18, 18))
plt.imshow(tweet1_wc, interpolation='bilinear')
plt.axis('off')
plt.show()

stemmer = SnowballStemmer("english", ignore_stopwords=True)
text_1 = df_2["tweet"].to_csv()
tokenizer = RegexpTokenizer(r"\w+")
word_tokens = tokenizer.tokenize(text_1)
filtered_sentence = [w for w in word_tokens if not w in stopwords]

filtered_sentence = []

for w in word_tokens:
    if w not in stopwords:
        lemmatizer = WordNetLemmatizer()
        w = lemmatizer.lemmatize(w)
        w = stemmer.stem(w)
        filtered_sentence.append(w)

split_it = str(filtered_sentence).split()

from collections import Counter
Counter = Counter(split_it)
most_occur = Counter.most_common(60)

frequentwords = pd.DataFrame(most_occur).rename(columns={0:"words", 1:"Frequencies"}).\
                sort_values(by="Frequencies").reset_index(drop=True)

frequentwords = frequentwords.drop(index=[6, 7, 15, 26, 41, 47, 56, 57]).reset_index(drop=True)
print(frequentwords)

fig = px.bar(frequentwords, x="Frequencies", y="words", orientation="h", height=1000,
            title="Most frequent words in the final days of Mr. Trump on twitter")
fig.show()


