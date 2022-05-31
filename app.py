#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from sklearn.model_selection import train_test_split
from audio_feature.audio_featurizer import extract_features, final



# header of the page

html_temp = """
    <div style ="background-color:powderblue;padding:13px">
    <h1 style ="color:black;text-align:center;">Predicting Hit Songs Using Repeated Chorus </h1>
    </div>
    """
      
st.markdown(html_temp, unsafe_allow_html = True)



###################################################          Loading the Final Data   #####################################################



def main():
    # Importing the dataset
    data=pd.read_csv("FinalData.csv")
    df=pd.DataFrame(data)




    df1=df.iloc[:,5:] # contains all numeric columns



    # ### Outlier Removal



    feature = df1 # one or more.
    for cols in feature:
        Q1 = df1[cols].quantile(0.25)
        Q3 = df1[cols].quantile(0.75)
        IQR = Q3 - Q1
        dff = df1[((df1[cols] < (Q1 - 1.5 * IQR)) |(df1[cols] > (Q3 + 1.5 * IQR)))]
    f=dff.index
    new_df = df.iloc[f]





    x = new_df.iloc[:,5:]# feature columns
    y = new_df['Label'] # target column




    # separate dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=0)
    print(" Training data: ",X_train.shape)
    print(" Testing data:  ",X_test.shape)


    # ### Feature Selection



    from sklearn.feature_selection import mutual_info_classif
    mutual_info=mutual_info_classif(X_train,y_train)
    mutual_info=pd.Series(mutual_info)
    mutual_info.index=X_train.columns
    s=mutual_info.sort_values(ascending=False)[:25]
    s.plot.bar(figsize=(10,6))
    f=s.index




    x1 = new_df[f]
    y1 = new_df["Label"]



    # separate dataset into train and test
    X1_train, X1_test, y1_train, y1_test = train_test_split(
        x1,
        y1,
        test_size=0.3,
        random_state=0)
    print(" Training data: ",X1_train.shape)
    print(" Testing data:  ",X1_test.shape)




    from imblearn.over_sampling import SMOTE
    # transform the dataset
    oversample = SMOTE()
    X, y = oversample.fit_resample(X1_train, y_train)



    # separate dataset into train and test
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0)
    print(" Training data: ",X2_train.shape)
    print(" Testing data:  ",X2_test.shape)



    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    X_train_norm1 = scaler.fit_transform(X2_train)
    X_test_norm1 = scaler.transform(X2_test)


    ###########################################                     Model Development                  ########################################



    #### Random Forest Classifier

    from sklearn import metrics
    from sklearn.ensemble import RandomForestClassifier
    # creating a RF classifier
    clf = RandomForestClassifier() 
    # Training the model on the training dataset
    # fit function is used to train the model using the training sets as parameters
    clf.fit(X_train_norm1, y2_train)
    # performing predictions on the test dataset
    y_pred2 = clf.predict(X_test_norm1)
    y_pred3 = clf.predict(X_train_norm1)
    score = metrics.accuracy_score(y2_test, y_pred2)*100
    print("Test accuracy:", round(score,2))
    score = metrics.accuracy_score(y2_train, y_pred3)*100
    print("Training accuracy:", round(score,2))



    #### Cross validating Random Forest


    #evaluating the model using repeated k-fold cross-validation
    from numpy import mean
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score

    #prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=3, n_repeats=7, random_state=1)
    # evaluate model
    scores = cross_val_score(clf, X_train_norm1, y2_train, scoring='accuracy', cv=cv, n_jobs=-1)
    #report performance
    print('Training Accuracy:',round(mean(scores),2)*100)




    #####################################################                User Interface              ##########################################


    # user input file

    #file = st.file_uploader("Upload Audio Files", type=["wav"])


    # button modify

    st.markdown("""
    <style>
    div.stButton > button:first-child {
        border-radius: 20%;
        height: 4em;
        width: 8em; 
        background-color: #0099ff;
        color:#ffffff;
    }
    div.stButton > button:hover {
        border-radius: 20%;
        height: 4em;
        width: 8em; 
        background-color: #00ff00;
        color:#ff0000;
        }
    </style>""", unsafe_allow_html=True)


    # button alignment

    col1, col2, col3 = st.columns([1,0.5,1])

    col4, col5, col6 = st.columns([1,0.5,1])


    rad_file = st.sidebar.radio("Select the name of song", ["The Weeknd - Blinding Lights", "Beyoncé - XO"])

    if rad_file == "The Weeknd - Blinding Lights":

        if col2.button("PREDICT"):

            data = []
            columns_name = []
            ca = ['The Weeknd - Blinding Lights.wav']

            for i , chorus in enumerate(ca):
                data , columns_name = extract_features("The Weeknd - Blinding Lights.wav", chorus)
                c = final(data,columns_name)
                x2 = c[f]

                # transforming the data
                d1 =np.array(x2)
                data1 = scaler.transform(d1)
                hit_prediction = clf.predict(data1)


                if hit_prediction == 1:
                    st.write("The predicted class is popular")

                elif hit_prediction == 0:
                    st.write("The predicted class is unpopular")

    elif rad_file == "Beyoncé - XO":

        if col2.button("PREDICT"):

            data = []
            columns_name = []
            ca = [file.name]

            for i , chorus in enumerate(ca):
                data , columns_name = extract_features("Beyoncé - XO.wav", chorus)
                c = final(data,columns_name)
                x2 = c[f]

                # transforming the data
                d1 =np.array(x2)
                data1 = scaler.transform(d1)
                hit_prediction = clf.predict(data1)


                if hit_prediction == 1:
                    st.write("The predicted class is popular")

                elif hit_prediction == 0:
                    st.write("The predicted class is unpopular")


if __name__=='__main__':
    main()



# In[ ]:




