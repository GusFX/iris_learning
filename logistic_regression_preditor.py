import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# loading data
dataset = pd.read_csv("dados/iris.csv")
X = dataset.iloc[:,0:-1]
y = dataset['class']

# encoding class
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)

# creating and training the model
model = LogisticRegression()
model.fit(train_X, train_y)
predict = model.predict(test_X)

# accuracy test
score = accuracy_score(test_y, predict)
score *= 100
print(f'Pricsão do modelo é de {score:.2f}%')
