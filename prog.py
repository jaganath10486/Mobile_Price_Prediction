import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier


dataframe = pd.read_csv("./train.csv")
data_test=pd.read_csv('./test.csv')
data_test.drop("price_range", axis=1, errors="ignore", inplace=True)

choice = 1
Visuallation_chice = 1

#Filling the Nan values with mean of respective column
def DataCleaning():
    print("-------------------------------------------Data Cleaning Process ---------------------------------")
    print(dataframe.isnull())
    #Replacing Null Values with mean of respective Columns
    columns = dataframe.columns
    for col in columns : 
        dataframe[col].replace(to_replace=np.nan, value=dataframe[col].mean(), inplace=True)
    print(dataframe)    

#Reducing the Dimension from 20 to 13
def DimensionReduction(X_train, X_test):
    lst = []
    pca = PCA(n_components = 13)
    X_train = pca.fit_transform(X_train)
    lst.append(X_train)
    X_test = pca.transform(X_test)
    lst.append(X_test)
    return lst

#Preprocessing 
def StandardScaling(X_train, X_test):
    lst = []
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    lst.append(X_train)
    X_test = sc.transform(X_test)
    lst.append(X_test)
    return lst

def dataSplit():
    lst = []
    X=dataframe.drop('price_range',axis=1)
    lst.append(X)
    y=dataframe['price_range']
    lst.append(y)
    return lst

#Linear Regression
def linearRegression():
    lst = dataSplit()
    #test size as 33%
    X_train, X_test, y_train, y_test = train_test_split(lst[0], lst[1], test_size=0.33, random_state=101)
    
    lm = LinearRegression()
    lst1 = DimensionReduction(X_train, X_test)
    lm.fit(lst1[0],y_train)
    print("The Score For Linear Regression is : ", lm.score(lst1[1],y_test))
    print("The Intercept For Linear Regression is : ", lm.intercept_)
    print("The Coefficient For Linear Regression is : ", lm.coef_)


#KNN Model
def KNN():
    lst = dataSplit()
    X_train, X_test, y_train, y_test = train_test_split(lst[0], lst[1], test_size=0.33, random_state=101)
    
    knn = KNeighborsClassifier(n_neighbors=10)
    lst1 = DimensionReduction(X_train, X_test)
    knn.fit(lst1[0],y_train)
    print("The Score For KNN is : ",knn.score(lst1[1],y_test))
    
    #Predection Values of Test Set
    y_test=knn.predict(lst1[1])
    print("Predicated Ranges for the Test set are : ")
    print(y_test)

    #predecting the Test.csv file
    data_test1=data_test.drop('id',axis=1, inplace=False, errors = "ignore")
    pca = PCA(n_components=13)
    data_test1 = pca.fit_transform(data_test1)
    predicted_price=knn.predict(data_test1)
    data_test['price_range'] = predicted_price

    print("Data Set after Predection is : ")
    print(data_test)
    data_test.to_csv("./test.csv")

print("-------------------------------Mobile Price Predection ------------------------------")
DataCleaning()
while(choice != 5):
    print("1. Data Info ")
    print("2. Dataset Describe ")
    print("3. Data Visualization & Analysis ")
    print("4. Price Range Predection")
    print("5. Exit ")
    choice = int(input("Enter the Choice : "))
    if choice == 1:
        print(dataframe.info())
    elif choice == 2:
        print(dataframe.describe())
    elif choice == 3:
        print("1. Data Visualization & Analysis between Price and Ram ")
        print("2. Data Visualization & Analysis between Price and Internal Memory ")
        print("3. percent of Phones which support 4G")
        print("4. Data Visualization & Analysis between Price and Battery Power")
        print("5. Data Visualization & Analysis between Clock Speed and Battery Power")
        Visuallation_chice = int(input("Enter the Choice : "))
        if Visuallation_chice == 1:
            plt.scatter(dataframe['ram'], dataframe['price_range'])
            plt.xlabel("RAM")
            plt.ylabel("Price_Range")
            plt.title("RAM VS Price_Range")
            plt.show()
        elif Visuallation_chice == 2:
            plt.scatter(dataframe['int_memory'], dataframe['price_range'])
            plt.xlabel("Internal Memory")
            plt.ylabel("Price_Range")
            plt.title("Internal Memory VS Price_Range")
            plt.show()
        elif Visuallation_chice == 3:
            labels4g = ["4G-supported",'Not supported']
            values4g = dataframe['four_g'].value_counts().values
            fig1, ax1 = plt.subplots()
            ax1.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
            plt.show()
        elif Visuallation_chice == 4:
            plt.scatter(x=dataframe["battery_power"], y=dataframe["price_range"])
            plt.xlabel("Battery Power")
            plt.ylabel("Price_Range")
            plt.title("Battery Power VS Price_Range")
            plt.show()
        elif Visuallation_chice == 5:
            plt.scatter(x=dataframe["clock_speed"], y=dataframe["price_range"])
            plt.xlabel("Clock Speed")
            plt.ylabel("Price_Range")
            plt.title("Clock Speed VS Price_Range")
            plt.show()
    elif choice == 4:
        print("1. BY Using KNN")
        print("2. By using Linear Regression Model")
        ModelChoice = int(input("Enter the Choice : "))
        if ModelChoice == 1:
            KNN()
        elif ModelChoice == 2:
            linearRegression()
    elif (choice > 5 | choice <= 0):
        print("Invlid Choice!!! Enter again")        
