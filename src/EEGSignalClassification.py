# -*- coding: utf-8 -*-

"""
!pip install visualkeras

!pip install pycaret
!pip install Jinja2
!pip install markupsafe==2.0.1

pip install scikit-learn==0.23.2
"""

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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import visualkeras
from PIL import ImageFont

import warnings
warnings.filterwarnings('ignore')

"""#Reading Data"""

eeg_data= pd.read_csv('https://raw.githubusercontent.com/onrkrsy/EEGSignalClassification/master/src/Datas/Epileptic%20Seizure%20Recognition.csv')
eeg_data.head()

eeg_data['y'].value_counts()

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
# print(eeg_data.head())
eeg_data.head()

#checking missing value
print("Missing Values")
print('---'*20)
print(eeg_data.isnull().sum())
print('---'*20)
print(eeg_data["Unnamed"].value_counts)

#Info
print("Columns Info")
print(eeg_data.info())
print(eeg_data.describe())

#removing First column
eeg_data = eeg_data.drop('Unnamed', axis = 1)
print("New verison of dataset")
eeg_data.head()

# Convert dataframe to matrix. Row: data entries. Column: features
X = eeg_data.iloc[:,0:178].values
print(X.shape)
print(X)
y = eeg_data.iloc[:,178].values
print(y.shape)
print(y)

"""# Train & Test Split"""

#Splitting the Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42,shuffle=True)
X_train.shape,y_test.shape
# Feature Scaling - Normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

"""#Models
##Classical Machine Learning Algorithms
we ll use pycaret to create models. and tune hyperparameters  """
import numpy as np
from pycaret.classification import *

clf1 = setup(data = eeg_data, target = 'y', numeric_imputation = 'mean', silent = True)
best_model = compare_models()
#Results of compared models
classification_results = pull()
"""#Optimal set of hyperparameters"""
tune_model(best_model)

"""#CNN """
def ModelCNN():
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(178,1), padding='same'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(64, 5, activation='relu', padding='same'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 7, activation='relu', padding='same'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(256, 9, activation='relu', padding='same'))
    model.add(MaxPool1D(2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.01) #burası kaldırılabilir. Araştırma konusu.
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

cnn = ModelCNN()
cnn.summary()

"""#Train CNN"""

history = cnn.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=128, verbose=2)

# list all data in history
print(history.history.keys())

"""#Plots of metrics"""

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

y_prob = cnn.predict(X_test,verbose = 1)
y_prob

y_pred = np.where(y_prob > 0.5, 1, 0)
y_pred

# confusion_matrix
confusion_matrix(y_test, y_pred)

# confusion_matrix
df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2), range(2))
sns.set(font_scale=1.6) # for label size
sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={"size": 17}) # font size
plt.show()

# classification report
print(classification_report(y_test, y_pred, target_names=["0", "1"]))

# classification report plot
clf_report = classification_report(y_test,
                                   y_pred,
                                   labels=np.arange(10),
                                   target_names=list("01"),
                                   output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)

"""Model Draw"""

tf.keras.utils.plot_model(
cnn,
to_file="model.png",
show_shapes=True,
show_dtype=False,
show_layer_names=True,
rankdir="TB",
expand_nested=True,
dpi=96,
layer_range=None,
show_layer_activations=True,
)

"""Model Save"""

cnn.save('cnn_model')