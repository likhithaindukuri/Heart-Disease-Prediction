from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import requests
from PIL import Image, ImageTk
import io

main = tkinter.Tk()
main.title("Heart Disease Prediction") 
main.geometry("1300x1200")
main.state('zoomed')

def set_background():
    url = "https://healthmatters.nyp.org/wp-content/uploads/2022/07/ai-heart-hero.jpg" 
    response = requests.get(url)  
    if response.status_code == 200:
        img_data = response.content
        img = Image.open(io.BytesIO(img_data))
        img = img.resize((1300, 700), Image.LANCZOS)  
        bg_image = ImageTk.PhotoImage(img)
        bg_label = Label(main, image=bg_image)
        bg_label.image = bg_image  
        bg_label.place(x=0, y=0, relwidth=1, relheight=1) 

set_background()  

global dataset
global filename
global le
global X_train, X_test, y_train, y_test, labels, scaler, xg_cls, pca
global accuracy, precision, recall, fscore

def show_output_window(title, content):
    output_window = Toplevel(main)
    output_window.title(title)
    output_window.geometry("600x400")
    text_area = Text(output_window, wrap=WORD, font=("Arial", 12))
    text_area.insert(END, content)
    text_area.pack(expand=True, fill=BOTH)
    close_button = Button(output_window, text="Close", command=output_window.destroy)
    close_button.pack()

def upload():
    global filename, dataset, labels, X
    filename = filedialog.askopenfilename(initialdir="Dataset")
    content = filename + " loaded\n\n"
    dataset = pd.read_csv(filename)
    X = dataset.values[:,0:dataset.shape[1]-1]
    content += str(dataset) + "\n\n"
    labels = np.unique(dataset['Label'])
    content += "Labels : " + str(labels)
    show_output_window("Dataset Upload", content)

    attack = dataset.groupby('Label').size()
    plt.figure(figsize=(8, 5))
    attack.plot(kind="bar")
    plt.xlabel('Labels')
    plt.ylabel('Count')
    plt.title('Dataset Labels Found in Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def preprocessing():
    global dataset, X, Y, le, scaler
    le = LabelEncoder()
    dataset['Label'] = pd.Series(le.fit_transform(dataset['Label'].astype(str)))
    dataset.fillna(0, inplace=True)
    dataset = dataset.values
    Y = dataset[:, dataset.shape[1]-1]
    scaler = StandardScaler()
    X = scaler.fit_transform(dataset[:, 0:dataset.shape[1]-1])
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    content = "Dataset processing completed. Normalized dataset values\n\n" + str(X)
    show_output_window("Preprocessing", content)

def trainTestSplit():
    global X, Y
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=99)
    content = "Dataset train & test split where application used 80% dataset size for training and 20% for testing\n\n"
    content += "80% training records : " + str(X_train.shape[0]) + "\n"
    content += "20% testing records  : " + str(X_test.shape[0]) + "\n"
    show_output_window("Train/Test Split", content)

def runSVM():
    global X_train, X_test, y_train, y_test, svm_cls, labels
    global accuracy, precision, recall, fscore

    svm_cls = SVC(kernel='linear')
    svm_cls.fit(X_train, y_train)
    predict = svm_cls.predict(X_test)
    accuracy = accuracy_score(y_test, predict) * 100
    precision = precision_score(y_test, predict, average='macro') * 100
    recall = recall_score(y_test, predict, average='macro') * 100
    fscore = f1_score(y_test, predict, average='macro') * 100
    content = f"SVM Accuracy  :  {accuracy}\nSVM Precision : {precision}\nSVM Recall    : {recall}\nSVM FScore    : {fscore}\n"
    show_output_window("SVM Results", content)
    
    conf_matrix = confusion_matrix(y_test, predict)
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(labels)])
    plt.title("SVM Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.xticks(rotation=45) 
    plt.tight_layout()
    plt.show()

def graph():
    global accuracy, precision, recall, fscore
    
    plt.figure(figsize=(8, 5)) 
    
    plt.bar(['Accuracy', 'Precision', 'Recall', 'F-score'], 
            [accuracy, precision, recall, fscore], 
            color=['lightblue', 'lightcoral', 'lightgreen', 'plum'], alpha=0.9, edgecolor='black')

    plt.xlabel("Comparison Metrics", fontsize=12, fontweight='bold')
    plt.ylabel("Values (%)", fontsize=12, fontweight='bold')
    plt.title("SVM Performance Metrics", fontsize=14, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def predict():
    global svm_cls, scaler, labels
    filename = filedialog.askopenfilename(initialdir="Dataset")
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace=True)
    dataset = dataset.values[:, 0:dataset.shape[1]-1]
    X = scaler.transform(dataset)
    predict = svm_cls.predict(X)
    content = ""
    for i in range(len(predict)):
        content += f"Test Data = {dataset[i]} =====> Predicted As : {labels[int(predict[i])]}\n\n"
    show_output_window("Prediction Results", content)

def on_enter(event):
    event.widget.config(bg='#0D1B2A', fg='white')

def on_leave(event):
    event.widget.config(bg='SystemButtonFace', fg='black')

   
font = ('Arial', 16, 'bold')  
title = Label(main, text='Machine Learning for Real-Time Heart Disease Prediction', bg='#0D1B2A', fg='white', font=font)  
title.place(x=0, y=0, relwidth=1, height=60)  
title.config(anchor="center")

font1 = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload ECG Dataset", command=upload,width=30, height=1)
uploadButton.place(x=900, y=200) 
uploadButton.config(font=font1)

preButton = Button(main, text="Dataset Preprocessing", command=preprocessing, width=30, height=1)
preButton.place(x=900, y=260) 
preButton.config(font=font1)

trainButton = Button(main, text="Train & Test Split", command=trainTestSplit,width=30, height=1)
trainButton.place(x=900, y=320)  
trainButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM,width=30, height=1)
svmButton.place(x=900, y=380)  
svmButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph, width=30, height=1)
graphButton.place(x=900, y=440)  
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Heart Disease from Test Data", command=predict, width=30, height=1)
predictButton.place(x=900, y=500)  
predictButton.config(font=font1)

uploadButton = Button(main, text="Upload ECG Dataset", command=upload,width=30, height=1)
uploadButton.place(x=900, y=200) 
uploadButton.config(font=font1)

preButton = Button(main, text="Dataset Preprocessing", command=preprocessing, width=30, height=1)
preButton.place(x=900, y=260) 
preButton.config(font=font1)

trainButton = Button(main, text="Train & Test Split", command=trainTestSplit,width=30, height=1)
trainButton.place(x=900, y=320)  
trainButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM,width=30, height=1)
svmButton.place(x=900, y=380)  
svmButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph, width=30, height=1)
graphButton.place(x=900, y=440)  
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Heart Disease from Test Data", command=predict, width=30, height=1)
predictButton.place(x=900, y=500)  
predictButton.config(font=font1)

for btn in [uploadButton, preButton, trainButton, svmButton, graphButton, predictButton]:
    btn.bind("<Enter>", on_enter)
    btn.bind("<Leave>", on_leave)

main.mainloop()
