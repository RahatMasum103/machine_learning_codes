
# coding: utf-8

# In[1]:


#*
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import math 

# ##################  plot_confusion_matrix function declaration ########################################
def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Count ', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    plt.show()

#////////////////////////////////////////////////////////////////////////////////////////
def main():
    try:
        # read the data
        df = pd.read_csv("spam-unbalanced.csv")
        # get the inputting data
        input_dt = df.ix[:, 2:-1]
        # get the output data
        output_dt = df.ix[:, -1]

        print(df.head(20))

        # ********************************************************************
        # index for string type data
        index = [1,2,5,6]
        # index for numerical type data
        index1 = [0, 3, 4, 7,8,9,10]

        # To convert data of inputting string type into numerical type*******************
        for i in index:
            # get the colmumn data
            col_dt = input_dt.ix[:, i]
            # create the encoder to convert string type into numerical data
            encoder = preprocessing.LabelEncoder()
            # Fit the encoder to convert string type into numerical data
            encoder.fit(col_dt)
            # Convert the encoder
            encoded_values = encoder.transform(col_dt)
            # put the original data into encoded data
            input_dt.ix[:, i] = encoded_values

        # To normalize data of inputing value type *******************
        for i in index1:
            #get the colmumn data
            col_dt = np.array(input_dt.ix[:, i])
            # convert the 1D data into 2D to calculate the result
            col_dt1 =[[x] for x in col_dt]
            # create the encoder to normalize data
            data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # implement the result
            data_scaled = data_scaler.fit_transform(col_dt1)
            input_dt.ix[:, i] = data_scaled
        for i in index:
            #get the colmumn data
            col_dt = np.array(input_dt.ix[:, i])
            # convert the 1D data into 2D to calculate the result
            col_dt1 =[[x] for x in col_dt]
            # create the encoder to normalize data
            data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            # implement the result
            data_scaled = data_scaler.fit_transform(col_dt1)
            input_dt.ix[:, i] = data_scaled  

        #********************************************************************
        print("-------------- modified input data -------------------")
        print(input_dt.head(20))

        # To convert output data into value type to find the result *******************
        # create the encoder to convert string type into numerical data
        encoder = preprocessing.LabelEncoder()
        # Fit the encoder to convert string type into numerical data
        encoder.fit(output_dt.loc[:])
        # Convert the encoder
        encoded_values = encoder.transform(output_dt)
        # put the original data into encoded data
        output_dt.loc[:] = encoded_values

        print("-------------- modified Label -------------------")
        print(output_dt.head(20))

        # show the distribution of the input attributes ****************
        input_dt.plot(kind='box', subplots=True, layout=(3, 4), sharex=False, sharey=False)
        plt.show('Distribution of the input attributes')

        # show the histogram of the input attributes ***************
        input_dt.hist()
        plt.title('Histogram of the input attributes')
        plt.show()
        output_dt.hist()
        plt.title('Histogram of the output attributes')
        plt.show()

        #****** to split the data into taining and testing data to train the a neural network **************
        # convert inputing and outputing data to array type to estimate the result
        input_dt = np.array(input_dt)
        output_dt = np.array(output_dt)

        # split the data into taining and testing data
        trainX, testX, trainY, testY = train_test_split(input_dt,output_dt,test_size = 0.40, random_state = 42)

        # *** To create and train a network**********************
        # to create Multi-layer neural network
        clf_MLP = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(5,), random_state=1)

        # to train Multi-layer neural network with training data
        clf_MLP.fit(trainX, trainY)

        # to find the output by Multi-layer neural network with test data
        y_pred_MLP = clf_MLP.predict(testX)

        # *** To find the confusion  ****************************************

        # find confusion matrix on testing data
        confusion = confusion_matrix(testY, y_pred_MLP)
        print("----- consion matrix ----------")
        print(confusion)

        # show confusion matrix on testing data as the graph
        plot_confusion_matrix(testY, y_pred_MLP)

        # *** To find another parameter for accuracy dicision *********

        print('--------- Parameters-------------------------')
    

        # To find the Precision value
        Precision = confusion[1, 1] / sum(confusion[ :,1])
        print('Precision  is %0.4f' % Precision)

        # To find the Sensitivity value
        #Sensitivity = confusion[0, 1] / sum(confusion[:, 1])
        #print('Sensitivity  is %0.4f' % Sensitivity)

        # To find the Recall value
        Recall = confusion[1, 1] / sum(confusion[1,:])
        print('Specificity  is %0.4f' % Recall)

        # To find the Specificity value
        #print('Recall  is %0.4f' % Sensitivity)

        # To find the F_score value
        F_score = 2 * Precision * Recall / (Precision + Recall)
        print('F_score  is %0.4f' % F_score)

        # To find the Accuracy value
        Accuracy = (confusion[1, 1] + confusion[0, 0]) / (confusion[1, 1] + confusion[0, 0]+ confusion[1, 0] + confusion[0, 1])
        print('Accuracy  is %0.4f'% Accuracy)
        
        # TO find Matthews correlation coefficient
        MCC = ((confusion[1, 1] * confusion[0, 0])-(confusion[0, 1])*(confusion[1, 0])) / (math.sqrt((confusion[1, 1]+confusion[0, 1])*(confusion[1, 1]+confusion[1, 0]) *(confusion[0, 0]+confusion[0, 1]) *(confusion[0, 0]+confusion[1, 0])))
        print ('MCC  is %0.4f'% MCC)
    except ValueError:
        print(ValueError)



##################################################
if __name__=="__main__":
    main()

