

**Epileptic Seizure Recognition**

According to the World Health Organization (WHO), epilepsy is a common neurological disorder that affects people of all ages. It is estimated that about 50 million people worldwide have epilepsy [1]. Recognizing seizures automatically is important for helping neurologists and other healthcare providers quickly diagnose patients and prescribe necessary treatments, if applicable. And Predicting seizures in advance could also help to prevent accidents and injuries that can occur during a seizure, such as falls or car accidents.

An EEG (electroencephalogram) is a test that measures and records the electrical activity of the brain. The test involves attaching small, flat metal discs (electrodes) to the scalp, which pick up the brain's electrical signals.

**Dataset** 

[Original Dataset Bonn\[2\] ](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/89c376b3-4ed7-41e9-8571-7ee14dc5c9c0/Andrzejak-PhysicalReviewE2001.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221219%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221219T170300Z&X-Amz-Expires=86400&X-Amz-Signature=b625d6506ac5a8466398b1e85e0dc3f93170577302a9ea318315dd3212d18569&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Andrzejak-PhysicalReviewE2001.pdf%22&x-id=GetObject) 

The Bonn dataset is collected under the supervision of the University of Bonn and consists of **five sets** of EEG recordings, where the first two sets (**A and B**) are captured from **healthy subjects**, and the other three sets (**C, D, and E)** are captured from five brain surgery candidates. Sets A and B vary in the state of the healthy subjects during the recording session with their **eyes open (set A)** and **closed (set B).** **Sets C and D** are EEG recordings in the **interictal state** from two different brain regions: the hippocampal (set C) and an epileptogenic zone (set D), whereas set E contains only **ictal state recordings.** Each set consists of **100 single-channel** EEG recordings with a duration of **23.6** s each, stored in textual file format. All the segments are preprocessed using a band-pass filter with a 0.53 Hz to 40 Hz cut-off frequency[3].
![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 002](https://user-images.githubusercontent.com/11960564/213981800-0be20f4e-7f99-4944-8e14-554518cf3f44.png)
![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 003](https://user-images.githubusercontent.com/11960564/213981843-0f6205b0-57b7-4830-9268-7347361fe3a8.png)

*Figure 1 Methods for measuring EEG signals*



[Re-structured/reshaped version on Kaggle\[4\]](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition) 

The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 txt with each has 4097 data points for 23.5 seconds.

That is divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time. So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}.  The response variable is y in column 179, the Explanatory variables X1, X2, …, X178 y contains the category of the 178-dimensional input vector. 

![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 004](https://user-images.githubusercontent.com/11960564/213981894-41f9c331-4392-4f46-8213-27637ec05ea7.png)

*Figure 2 Dataset (11500 row x 180 column)*

![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 005](https://user-images.githubusercontent.com/11960564/213981914-75e35e8c-0a62-413d-a58d-4d37a97e5a63.png)

*Figure 3 Data Categories*

Figure 3 shows that every category has 2300 rows.

Some samples from different categories.
![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 006](https://user-images.githubusercontent.com/11960564/213981959-d9734f3c-3eb8-483a-819c-3e598736530c.png)

*Figure 4 One sample line of data from different categories*

**Data Pre-Processing**
**


Data pre-processing is an important step in any machine learning or data analysis project. It involves cleaning and transforming the raw data into a format that can be used by the model. This step is crucial for ensuring the accuracy and reliability of the final results. Some common data pre-processing techniques include removing missing or duplicate data, normalizing or standardizing the data, and transforming categorical variables into numerical ones.
**


**Checking Missing Data**

By checking with the **isnull().sum()** function, we can understand that **there is no missing** values.
![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 007](https://user-images.githubusercontent.com/11960564/213981992-46e01f60-c17e-49bf-a850-cb2de3ab69df.png)

*Figure 5 Checking Missing Values*

**Data Cleaning**

Removing unnecessary data is a process of eliminating data that is not needed for the specific analysis or model being used. This can include data that is duplicated, irrelevant, or contains errors. By removing this data, the analysis or model can be made more efficient, accurate, and reliable. 

**The first column** contains a cookie for each row. There will be no meaningful data in the classification process. So we **remove** the first column.

Then the target variable was converted into binary classes (1 for seizure and 0 for no seizure).
![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 008](https://user-images.githubusercontent.com/11960564/213982019-e53dda9c-1304-4b54-8e3b-912101a77ed0.png)

*Figure 6 Unnamed column is removed*

**Classification**

**Classical Machine Learning Algorithms**

PyCaret is an open-source, low-code machine learning library in Python that allows users to go from preparing the data to deploying machine learning models with ease. It is a library that streamlines the process of training and deploying machine learning models by providing a set of pre-built functions that automate various tasks such as data pre-processing, model training, tuning, and deployment. It is used to help data scientists, analysts, and engineers to save time and effort by automating repetitive tasks in the machine learning workflow.

The PyCaret library is used for to create classic Machine Learning models. With this library, all models are tested with a short command and the results are presented in a table form. 

**Light Gradient Boosting** model has best accuracy rate with **0,9725.** 

![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 009](https://user-images.githubusercontent.com/11960564/213982041-ce67f85c-91ac-4044-b4a0-279a5d6bc910.png)

*Figure 7 Experimantal Results for Machine Learning Algorithms*

**Optimal set of hyperparameters** 

Tunemodel function in PyCater library that is used to find the optimal set of hyperparameters for a given machine learning model. Hyperparameters are parameters that are not learned from the data, but rather set before training the model. Examples of common hyperparameters include the learning rate, number of hidden layers, and regularization term.
![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 010](https://user-images.githubusercontent.com/11960564/213982062-e453142b-7d84-42b4-a759-43071f3d0a5f.png)

Tunemodel  function is used to optimize the performance of a machine learning model by finding the best set of hyperparameters for the given dataset. With hyperparameter optimizing the accuracy value is improved from **0,9725** to **0,9839**

**CNN Model** 

1D Convolutional Neural Networks (CNNs) can be used when the input data is a sequence of one-dimensional values, such as time series data, audio signals, or text data. They are particularly useful for extracting features from the data and recognizing patterns, and are often used for tasks such as time series prediction, audio classification, and natural language processing. So a CNN model is used to create a new model. 
![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 011](https://user-images.githubusercontent.com/11960564/213982095-08ee171e-3f8f-49ef-98e9-7ca922bda27b.png)

*Figure 8 Created CNN Model*

![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 012](https://user-images.githubusercontent.com/11960564/213982136-813dafac-24ae-4c4d-8a01-78f8d7023486.png)

*Figure 9 Created CNN Model Features*

![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 013](https://user-images.githubusercontent.com/11960564/213982171-ae11e67f-15cb-4e07-9c26-428e93b07c2c.png)

*Figure 10 History for Accuracy*


![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 014](https://user-images.githubusercontent.com/11960564/213982200-e4abc1d2-fc33-4bd1-b0e4-490d4acfaf8f.png)

*Figure 11  History for Loss*

![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 015](https://user-images.githubusercontent.com/11960564/213982219-387ac8b4-5ec0-4a6f-a86a-9c388e44e5a2.png)

*Figure 12 Classification Report*

![Aspose Words 4f30531d-e2a6-4733-a6b2-be1369365238 016](https://user-images.githubusercontent.com/11960564/213982236-ddda6be4-3d2c-40bb-b383-e798b35c4e29.png)

*Figure 13 Confusion Matrix of created CNN model.*


**Conclusion**

In this work, an EEG signal dataset was used to classify whether the data represents a seizure or not. A variety of classical machine learning algorithms were tested using PyCater to evaluate their performance. The best model was found to be **Light Gradient Boosting model**, which achieved an accuracy rate of **0.9725.** By using hyperparameter tuning, the performance of this model was improved to **0.9839.** After that, a 1D CNN model was created, which resulted in an even higher accuracy rate of **0.99**. This indicates that the 1D CNN model was the best model among the tested models, providing accurate classification of the EEG signal dataset with seizure or not-seizure labels.

**References**

[1] Epilepsy. Available online: <https://www.who.int/news-room/fact-sheets/detail/epilepsy>

[2] Andrzejak RG, Lehnertz K, Rieke C, Mormann F, David P, Elger CE (2001) Indications of nonlinear deterministic and finite dimensional structures in time series of brain electrical activity: Dependence on recording region and brain state,Phys. Rev. E, 64, 061907

[3] Nafea, Mohamed & Ismail, Zool. (2022). Supervised Machine Learning and Deep Learning Techniques for Epileptic Seizure Recognition Using EEG Signals—A Systematic Literature Review. Bioengineering. 9. 781. 10.3390/bioengineering9120781.

[4] Available online: <https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition>

[5] Available online:  <https://pycaret.gitbook.io/docs/>

