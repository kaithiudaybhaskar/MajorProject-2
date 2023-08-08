from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import webbrowser
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

global filename

#global X,Y
global dataset
global main
global text
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test, predict_cls
global classifier

main = tkinter.Tk()
main.title("Comparative Study of Machine Learning Algorithms for Fraud Detection in Blockchain") #designing main screen
main.geometry("1000x1200")

 
#fucntion to upload dataset
def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")
    dataset = pd.read_csv(filename)
    text.insert(END,"Dataset before preprocessing\n\n")
    text.insert(END,str(dataset.head()))
    text.update_idletasks()
    label = dataset.groupby('FLAG').size()
    label.plot(kind="bar")
    plt.title("Blockchain Fraud Detection Graph 0 means Normal & 1 means Fraud")
    plt.show()
    

#function to perform dataset preprocessing
def trainTest():
    global X,Y
    global dataset
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    #replace missing values with 0
    dataset.fillna(0, inplace = True)
    Y = dataset['FLAG'].values.reshape(-1,1)
    Y = Y.ravel()  # convert Y to 1D array
    dataset = dataset.values
    X = dataset[:,4:dataset.shape[1]-2].reshape(-1, dataset.shape[1]-6)
    X = normalize(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X = X[0:5000]
    Y = Y[0:5000]     
    print(Y)
    print(X)
    text.insert(END,"Dataset after features normalization\n\n")
    text.insert(END,str(X)+"\n\n")
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total features found in dataset: "+str(X.shape[1])+"\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Dataset Train and Test Split\n\n")
    text.insert(END,"80% dataset records used to train ML algorithms : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used to test ML algorithms : "+str(X_test.shape[0])+"\n")



def calculateMetrics(algorithm, predict, y_test):
    y_test = y_test.ravel()  # convert y_test to 1D array
    a = str(accuracy_score(y_test,predict)*100)[0:5]
    p = str(precision_score(y_test, predict,average='macro') * 100)[0:5]
    r = str(recall_score(y_test, predict,average='macro') * 100)[0:5]
    f = str(f1_score(y_test, predict,average='macro') * 100)[0:5]
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    text.insert(END,algorithm+" Accuracy  :  "+a+"\n")
    text.insert(END,algorithm+" Precision : "+p+"\n")
    text.insert(END,algorithm+" Recall    : "+r+"\n")
    text.insert(END,algorithm+" FScore    : "+f+"\n\n")


def runLogisticRegression():
    global X,Y, X_train, X_test, y_train, y_test
    global accuracy, precision,recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    lr = LogisticRegression() 
    lr.fit(X, Y) 
    predict = lr.predict(X_test)
    calculateMetrics("Logistic Regression", predict, y_test)

def runMLP():
    mlp = MLPClassifier() 
    mlp.fit(X_train, y_train) 
    predict = mlp.predict(X_test)
    calculateMetrics("MLP", predict, y_test)
    

def runNaiveBayes():
    cls = GaussianNB() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    calculateMetrics("Naive Bayes", predict, y_test)

def runAdaBoost():
    cls = AdaBoostClassifier() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    calculateMetrics("AdaBoost", predict, y_test)
       
    
def runDT():
    global predict_cls
    cls = DecisionTreeClassifier() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    calculateMetrics("Decision Tree", predict, y_test)
    

def runSVM():
    cls = svm.SVC() 
    cls.fit(X_train, y_train) 
    predict = cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)
    

def runRF():
    global predict_cls
    rf = RandomForestClassifier() 
    rf.fit(X_train, y_train) 
    predict = rf.predict(X_test)
    predict_cls = rf
    calculateMetrics("Random Forest", predict, y_test)

def predict():
    global predict_cls
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    dataset = dataset.values
    X = dataset[:,4:dataset.shape[1]-2]
    X1 = normalize(X)
    prediction = predict_cls.predict(X1)
    print(prediction)
    for i in range(len(prediction)):
        if prediction[i] == 0:
            text.insert(END,"Test DATA : "+str(X[i])+" ===> PREDICTED AS NORMAL\n\n")
        else:
            text.insert(END,"Test DATA : "+str(X[i])+" ===> PREDICTED AS FRAUD\n\n")
    
    
def runDeepNetwork():
    global X_train, X_test, y_train, y_test
    global accuracy, precision, recall, fscore
    global classifier
    text.delete('1.0', END)
    classifier = Sequential()
    # Add layers to the model
    classifier.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1]))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.fit(X_train, y_train, epochs=20, batch_size=128, verbose=1)
    predicted_probabilities = classifier.predict(X_test)
    predicted_classes = (predicted_probabilities > 0.5).astype('int32')
    calculateMetrics("Deep Neural Network", predicted_classes, y_test)


    

def graph():
    if len(accuracy) < 8 or len(precision) < 8 or len(recall) < 8 or len(fscore) < 8:
        print("Error: The lists do not have enough elements.")
        return 


    
    output = "<html><body><table align=center border=1><tr><th>Algorithm Name</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
    output+="<th>FSCORE</th></tr>"
    output+="<tr><td>Logistic Regression Algorithm</td><td>"+str(accuracy[0])+"</td><td>"+str(precision[0])+"</td><td>"+str(recall[0])+"</td><td>"+str(fscore[0])+"</td></tr>"
    output+="<tr><td>MLP Algorithm</td><td>"+str(accuracy[1])+"</td><td>"+str(precision[1])+"</td><td>"+str(recall[1])+"</td><td>"+str(fscore[1])+"</td></tr>"
    output+="<tr><td>Naive Bayes Algorithm</td><td>"+str(accuracy[2])+"</td><td>"+str(precision[2])+"</td><td>"+str(recall[2])+"</td><td>"+str(fscore[2])+"</td></tr>"
    output+="<tr><td>AdaBoost Algorithm</td><td>"+str(accuracy[3])+"</td><td>"+str(precision[3])+"</td><td>"+str(recall[3])+"</td><td>"+str(fscore[3])+"</td></tr>"
    output+="<tr><td>Decision Tree Algorithm</td><td>"+str(accuracy[4])+"</td><td>"+str(precision[4])+"</td><td>"+str(recall[4])+"</td><td>"+str(fscore[4])+"</td></tr>"
    output+="<tr><td>SVM Algorithm</td><td>"+str(accuracy[5])+"</td><td>"+str(precision[5])+"</td><td>"+str(recall[5])+"</td><td>"+str(fscore[5])+"</td></tr>"
    output+="<tr><td>Random Forest Algorithm</td><td>"+str(accuracy[6])+"</td><td>"+str(precision[6])+"</td><td>"+str(recall[6])+"</td><td>"+str(fscore[6])+"</td></tr>"
    output+="<tr><td>Deep Neural Network Algorithm</td><td>"+str(accuracy[7])+"</td><td>"+str(precision[7])+"</td><td>"+str(recall[7])+"</td><td>"+str(fscore[7])+"</td></tr>"
    output+="</table></body></html>"
    f = open("table.html", "w")
    f.write(output)
    f.close()
    webbrowser.open("table.html",new=1)

    df = pd.DataFrame({'i': [1,2,3,4],
                       'Parameters': [['Precision', 'Accuracy', 'Fscore', 'Recall'],['Precision', 'Accuracy', 'Fscore', 'Recall'],['Precision', 'Accuracy', 'Fscore', 'Recall'],['Precision', 'Accuracy', 'Fscore', 'Recall'],],
                       'Logistic Regression': [precision[0],accuracy[0],fscore[0],recall[0]],
                       'MLP': [precision[1],accuracy[1],fscore[1],recall[1]],
                       'Naive Bayes': [precision[2],accuracy[2],fscore[2],recall[2]],
                       'AdaBoost': [precision[3],accuracy[3],fscore[3],recall[3]],
                       'Decision Tree': [precision[4],accuracy[4],fscore[4],recall[4]],
                       'SVM': [precision[5],accuracy[5],fscore[5],recall[5]],  
                       'Random Forest': [precision[6],accuracy[6],fscore[6],recall[6]],  
                       'Deep Neural Network': [precision[7],accuracy[7],fscore[7],recall[7]],  
                       })
    df.pivot(index='i', columns='Parameters', values=['Logistic Regression', 'MLP', 'Naive Bayes', 'AdaBoost', 'Decision Tree','SVM','Random Forest','Deep Neural Network']).plot(kind='bar')
    plt.show()
    
    """df =pd.DataFrame([['Logistic Regression','Precision',precision[0]],['Logistic Regression','Recall',recall[0]],['Logistic Regression','F1 Score',fscore[0]],['Logistic Regression','Accuracy',accuracy[0]],
                       ['MLP','Precision',precision[1]],['MLP','Recall',recall[1]],['MLP','F1 Score',fscore[1]],['MLP','Accuracy',accuracy[1]],
                       ['Naive Bayes','Precision',precision[2]],['Naive Bayes','Recall',recall[2]],['Naive Bayes','F1 Score',fscore[2]],['Naive Bayes','Accuracy',accuracy[2]],
                       ['AdaBoost','Precision',precision[3]],['AdaBoost','Recall',recall[3]],['AdaBoost','F1 Score',fscore[3]],['AdaBoost','Accuracy',accuracy[3]],
                       ['Decision Tree','Precision',precision[4]],['Decision Tree','Recall',recall[4]],['Decision Tree','F1 Score',fscore[4]],['Decision Tree','Accuracy',accuracy[4]],
                       ['SVM','Precision',precision[5]],['SVM','Recall',recall[5]],['SVM','F1 Score',fscore[5]],['SVM','Accuracy',accuracy[5]],
                       ['Random Forest','Precision',precision[6]],['Random Forest','Recall',recall[6]],['Random Forest','F1 Score',fscore[6]],['Random Forest','Accuracy',accuracy[6]], 
                       ['Deep Neural Network','Precision',precision[7]],['Deep Neural Network','Recall',recall[7]],['Deep Neural Network','F1 Score',fscore[7]],['Deep Neural Network','Accuracy',accuracy[7]], 
                      ],columns=['Parameters','Algorithms','Value'])  
    df.plot(kind='bar')
    plt.show() """


   


font = ('arial', 16, 'bold')
title = Label(main, text='Comparative Study of Machine Learning Algorithms for Fraud Detection in Blockchain')
title.config(bg='#189AB4', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('Arial', 12, 'bold')
text=Text(main,height=20,width=125)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('arial', 13, 'bold')
uploadButton = Button(main, text="Upload & Preprocess Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

traintestButton = Button(main, text="Generate Train & Test Model", command=trainTest)
traintestButton.place(x=330,y=550)
traintestButton.config(font=font1) 

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLogisticRegression)
lrButton.place(x=630,y=550)
lrButton.config(font=font1)

mlpButton = Button(main, text="Run MLP Algorithm", command=runMLP)
mlpButton.place(x=950,y=550)
mlpButton.config(font=font1)

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNaiveBayes)
nbButton.place(x=50,y=600)
nbButton.config(font=font1) 

adaboostButton = Button(main, text="Run AdaBoost Algorithm", command=runAdaBoost)
adaboostButton.place(x=330,y=600)
adaboostButton.config(font=font1)

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDT)
dtButton.place(x=630,y=600)
dtButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=950,y=600)
svmButton.config(font=font1)


rfButton = Button(main, text="Run Random Forest Algorithm", command=runRF)
rfButton.place(x=50,y=650)
rfButton.config(font=font1)


dnButton = Button(main, text="Run Deep Network Algorithm", command=runDeepNetwork)
dnButton.place(x=330,y=650)
dnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=630,y=650)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Fraud ", command=predict)
predictButton.place(x=950,y=650)
predictButton.config(font=font1)

main.config(bg='#D4F1F4')
main.mainloop()



