import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv1D, MaxPool1D, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

eeg_data= pd.read_csv('Datas/Epileptic Seizure Recognition.csv')
print(eeg_data.head())
#print(eeg_dataframe.tail())
#print(eeg_dataframe.describe())
#print(eeg_dataframe.isnull().sum())

print(eeg_data['y'].value_counts());
def hist(df,plt):
  plt.hist(df[df["y"]==1]["y"],label="Seizure" )
  plt.hist(df[df["y"]==2]["y"],label="Not Seizure - epileptogenic zone")
  plt.hist(df[df["y"]==3]["y"],label="Not Seizure - healty area")
  plt.hist(df[df["y"]==4]["y"],label="Not Seizure - eyes closed")
  plt.hist(df[df["y"]==5]["y"],label="Not Seizure - eyes open")
  plt.legend(loc='lower right')
  plt.show()
#plot of categories
hist(eeg_data,plt)
#Sample data
X=eeg_data.values
X=X[:,1:-1]
plt.figure(figsize=(12,8))
plt.plot(X[1,:],label='Seizure')
plt.plot(X[7,:],label='Not Seizure - epileptogenic zone')
plt.plot(X[12,:],label='Not Seizure - healty area')
plt.plot(X[0,:],label='Not Seizure - eyes closed')
plt.plot(X[2,:],label='Not Seizure - eyes open')
plt.legend()
plt.show()



# make seizure and not seizure classes 1 is seizure and others  are 0 not seizure
dic = {5: 0, 4: 0, 3: 0, 2: 0, 1: 1}
eeg_data['y'] = eeg_data['y'].map(dic)
print("Seizure and Not Seizure Counts:")
print(eeg_data['y'].value_counts());
print("New dataset with binary class")
print(eeg_data.head())

#checking missing value
print("Missing Values")
print(eeg_data.isnull().sum())
print(eeg_data["Unnamed"].value_counts)

#Info
print("Columns Info")
print(eeg_data.info())
print(eeg_data.describe())

#removing unnecessary column
eeg_data = eeg_data.drop('Unnamed', axis = 1)
print("New verison of dataset")
print(eeg_data.head())

# Convert dataframe to matrix. Row: data entries. Column: features
X = eeg_data.iloc[:,0:178].values
print(X.shape)
print(X)
y = eeg_data.iloc[:,178].values
print(y.shape)
print(y)


#Splitting the Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42,shuffle=True)
X_train.shape,y_test.shape
# Feature Scaling - Normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""
# Support Vector Classifier
from sklearn.svm import SVC
# Create classifier instance
svc_clf = SVC()
# Fit to train data
svc_clf.fit(X_train, y_train)
# Evaluate on train data
train_acc_svc =  svc_clf.score(X_train, y_train)
print("Train accuracy of SVM:", round(train_acc_svc * 100, 2), "%") # Round the acc to 2 number after the decimal point
# Evaluate on test data
test_acc_svc = svc_clf.score(X_test, y_test)
print("Test accuracy of SVM:", round(test_acc_svc *100, 2), "%") # Round the acc to 2 number after the decimal point

"""

def ModelANN():
    model = Sequential()
    model.add(Conv1D(256, 3, activation='relu', input_shape=(178,1), padding='same'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



#classifiers
models = [LogisticRegression(), SVC(),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          GradientBoostingClassifier(),
          KNeighborsClassifier()]

#Check the correctness of list of classifiers and also
model_name = [type(model).__name__ for model in models]
print(model_name)

# all parameters are not specified are set to their defaults
def classifiers(models):
    columns = ['Score', 'Predictions']
    df_result = pd.DataFrame(columns=columns, index=[type(model).__name__ for model in models])

    for model in models:
        clf = model
        print('Initialized classifier {} with default parameters \n'.format(type(model).__name__))
        clf.fit(X_train, y_train)
        #make a predicitions for entire data(X_test)
        predictions = clf.predict(X_test)
        # Use score method to get accuracy of model
        score = clf.score(X_test, y_test)
        print('Score of classifier {} is: {} \n'.format(type(model).__name__, score))
        df_result['Score']['{}'.format(type(model).__name__)] = str(round(score * 100, 2)) + '%'
        df_result['Predictions']['{}'.format(type(model).__name__)] = predictions
    return df_result

#print(classifiers(models))

ann = ModelANN()
ann.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=128, verbose=2)
#ortlama sonuç?