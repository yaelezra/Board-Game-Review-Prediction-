# importing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# loading data
data = pd.read_csv('games.csv')

# some information about the data ...
data_des = data.describe()

na_data = data.isna().sum()

# one hot encoding for data - type
plt.hist(data['type'])
plt.title('Histogram of the "type" Feature')
plt.show()


data_dummies = pd.get_dummies(data['type'], prefix='type')
data = data.drop(['type'], axis=1)
data = pd.concat([data, data_dummies], axis=1)

# drop rows with nas
data = data.dropna(axis=0)

# choosing the target variable - average rating
features_data = data.copy()
features_data = features_data.drop(['average_rating'], axis=1)

labels = data['average_rating']

# plot histogram of target variable
plt.hist(labels)
plt.title('Histogram of Target variable')
plt.show()

# the mean values of the games with zero rating
data_zero = features_data[labels == 0].mean()

# We will keep only the games that were rated by users and plot the histogram again :)
data = data[data['users_rated'] > 0]

features_data = data.copy()
features_data = features_data.drop(['average_rating'], axis=1)

labels = data['average_rating']

plt.hist(labels)
plt.title('Histogram of Target variable')
plt.show()

# correlation matrix between features
corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corr_matrix)
plt.show()

# dropping the name feature (doesn't have an affect)
features_data = features_data.drop(['name'], axis=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(features_data, labels, test_size=0.2, random_state=0)


# Scaling data
scaler = StandardScaler()
train_set = scaler.fit_transform(X_train)
test_set = scaler.transform(X_test)

# training the model
model = LinearRegression()
model.fit(train_set, y_train)

# Computing train error
model_train_error = mean_squared_error(y_train, model.predict(train_set))

# Computing the test error
model_test_error = mean_squared_error(y_test, model.predict(test_set))

