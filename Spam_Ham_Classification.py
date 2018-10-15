import nltk
import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
#nltk.download_shell()
messages=[line.rstrip() for line in open("SMSSpamCollection")]
print(messages[0])
for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print("\n")

messages=pd.read_csv("SMSSpamCollection",sep="\t",names=["label","message"])
print(messages.head())


#adding length column
messages["length"]=messages["message"].apply(len)
print(messages.head())

#visualizing the data
print(messages.describe())
print(messages.groupby("label").describe())
messages["length"].plot.hist(bins=100)
plt.show()
messages.hist(column="length",by="label",figsize=(10,8),bins=100)
plt.show()

#NLP
def text_process(msg):
    x = [char for char in msg if char not in string.punctuation]

    x = "".join(x)

    y=[]
    y = [word for word in x.split() if word.lower() not in stopwords.words("english")]
    return y

msg_train,msg_test,label_train,label_test=train_test_split(messages["message"],messages["label"],test_size=0.3)

pipeline=Pipeline([("bow",CountVectorizer(analyzer=text_process)),
                   ("tfidf",TfidfTransformer()),
                   ("Classifier",MultinomialNB())])
pipeline.fit(msg_train,label_train)
pred=pipeline.predict(msg_test)
print("confusion matrix :")
print(confusion_matrix(label_test,pred))
print("Classification report :")
print(classification_report(label_test,pred))






