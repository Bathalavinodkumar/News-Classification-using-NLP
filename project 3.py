
import nltk

nltk.download('punkt')

import pandas as pd

fake= pd.read_csv("C:\\Users\\Vinod\\Desktop\\Resume\\Fake.csv")
genuine = pd.read_csv("C:\\Users\\Vinod\\Desktop\\Resume\\true.csv")


print(fake)

print(fake.info())
print(genuine.info())


print(fake.head(20))
print(genuine.head())


print(fake.subject.value_counts())
print('\n')
print(genuine.subject.value_counts())


fake['target'] = 0
genuine['target'] = 1

print(fake.head())
print(genuine.head())

data = pd.concat([fake, genuine], axis=0)
data = data.reset_index(drop=True)
print(data)

data = data.drop(['subject', 'date', 'title'], axis =1 )
print(data.columns)
print(data)

from nltk.tokenize import word_tokenize
data['text']=data['text'].apply(word_tokenize)
print(data.head(10))


from nltk.stem.snowball import SnowballStemmer
porter = SnowballStemmer( 'english', ignore_stopwords=False )


def stem_it(text):
    return[porter.stem(word) for word in text]
data['text']=data['text'].apply(stem_it)
print(data.head(10))



def stop_it(t):
    dt=[word for word in t if len(word)>2]
    return dt
data['text'].apply(stop_it)
print(data['text'].head(10))
data['text']=data['text'].apply(' '.join)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.25)
display(X_train.head())
print('\n')
display(y_train.head())


from sklearn.feature_extraction.text import TfidfVectorizer
my_tfidf = TfidfVectorizer( max_df=0.7)
tfidf_train = my_tfidf.fit_transform(X_train)
tfidf_test = my_tfidf.transform(X_test)
print(tfidf_train)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_1 = LogisticRegression(max_iter=900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
cr1    = accuracy_score(y_test,pred_1)
print(cr1*100)


from sklearn.linear_model import PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)


PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
              early_stopping=False, fit_intercept=True, loss='hinge',
              max_iter=50, n_iter=None, n_iter_no_change=5, n_jobs=None,
              random_state=None, shuffle=True, tol=None,
              validation_fraction=0.1, verbose=0, warm_start=False)


y_pred = model.predict(tfidf_test)
accscore = accuracy_score(y_test, y_pred)
print('The accuracy of prediction is ',accscore*100)





