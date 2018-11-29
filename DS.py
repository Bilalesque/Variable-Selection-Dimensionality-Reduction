###############################################################################
#########################  Mohammad Bilal Shafique  ###########################
#########################          K152152          ###########################
#########################        Section GR2        ###########################
###############################################################################

import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import RFE

file1=open("Simple Linear Regression.txt","w+")
file2=open("Forward Selection.txt","w+")
file3=open("Linear SVR.txt","w+")
file4=open("Recursive Feature Elimination.txt","w+")
file5=open("PCA.txt","w+")

def Forward_Selection(dataset,dependent,label,v):
    
    chosen=[]
    bestscore,cscore=0.0,0.0
    remainder=set(dataset.columns)
    remainder.remove(dependent)
    temp=[]
    categorical=dependent
    while bestscore == cscore and remainder:
        pair=[]
        for considered in remainder:
            print('\nVariable Considered: %s' %considered)
            file2.write('\nVariable Considered: %s' %considered)
            if data[considered].dtype != object:    #Encoding categorical data
                equation="{} ~ {}".format(categorical,' + '.join(chosen+[considered]))
            else:
                if chosen!=[]:
                    equation="{} ~ {} + {}".format(categorical,' + '.join(chosen),'C('+considered+')')
                else:
                    equation="{} ~ {}".format(categorical,'C('+considered+')')
            print(equation)
            file2.write("\n %s" %equation)
            score=smf.ols(equation,dataset).fit().rsquared_adj
            pair.append((score,considered))
        pair.sort()
        bestscore,selected=pair.pop()
        if  bestscore > cscore:
            cscore=bestscore
            chosen.append(selected)
            remainder.remove(selected) 
            print("\nCurrent Score After Selecting %s: "%selected+str(cscore))
            file2.write("\nCurrent Score After Selecting %s: "%selected+str(cscore))
            print()
            file2.write("\n")
    for y in chosen:
        if temp != []:
            if data[y].dtype != object:
                temp="{} + {}".format(temp,y)
            else:
                #encoding categorical data
                temp="{} + {}".format(temp,'C('+y+')')
        else:
            if data[y].dtype != object:
                temp=y
            else:
               #encoding categorical data
                temp='C('+y+')'
    #Combining formula
    equation="{} ~ {} + 1".format(categorical,temp)
    model=smf.ols(equation,data).fit()
    rank=[]
    for i in range(0,len(chosen)):
        rank.append(0)
    count=0
    for i in range(0,len(chosen)):
        rank[count]=i
        temp=label[i]
        label[i]=label[count]
        label[count]=temp
        count+=1
    print("\nSelected Variables")
    file2.write("\nSelected Variables")
    for i in range(0,4):
        print(labels[i])
        file2.write("\n %s" %labels[i])
        
    Select=X[:,[rank[0],rank[1],rank[2],rank[3]]]
    print("\nData of Selected Features: \n%s" %Select)
    file2.write("\nData of Selected Features: \n%s" %Select)
    model=smf.ols(equation,data).fit()
    
    print("\nFinal Equation After Forward Selection: %s" %model.model.formula)
    file2.write("\nFinal Equation After Forward Selection: %s" %model.model.formula)
    print("\nR^2 Adjusted After Forward Selection Model Fitting: %s" %model.rsquared_adj)
    file2.write("\nR^2 Adjusted After Forward Selection Model Fitting: %s" %model.rsquared_adj)
    print(model.summary())
    file2.write("\n%s" %model.summary())
    x_train,x_test,y_train,y_test=train_test_split(Select,v,test_size=0.67,random_state=0)
    Lmodel=LinearRegression()
    Lmodel.fit(x_train,y_train)
    prediction=Lmodel.predict(x_test)
    print("\nPrediction After Forward Selection: %s" %prediction)
    file2.write("\nPrediction After Forward Selection: %s" %prediction)
    score=r2_score(y_test,prediction)
    print("\nR^2 After Forward Selection: %s" %score)
    file2.write("\nR^2 After Forward Selection: %s" %score)
    return score

def LSVR(x_train,x_test,y_train,y_test):
    
    model=LinearSVR(random_state=0)
    model.fit(x_train,y_train)
    LinearSVR(C=1.0,dual=True,epsilon=0.0,fit_intercept=True,intercept_scaling=1.0,loss='epsilon_insensitive',max_iter=1000,random_state=0,tol=0.0001,verbose=0)
    print("\nLinear SVR Coefficients %s" %model.coef_)
    file3.write("\nLinear SVR Coefficients %s" %model.coef_)
    prediction=model.predict(x_test)
    print("\nPrediction With Linear SVR: %s" %prediction)
    file3.write("\nPrediction With Linear SVR: %s" %prediction)
    score=r2_score(y_test,prediction)
    print("\nR^2 After Linear SVR: %s" %score)
    file3.write("\nR^2 After Linear SVR: %s" %score)
    return score

def PrincipleComponents(K,x,x_train,x_test,y_train,y_test):
    
    pca=PCA(n_components=K)
    PC=pca.fit_transform(x)
    PCAdataframe=pd.DataFrame(data=PC,columns=['PC1','PC2','PC3','PC4'])
    ConcatComponents=pd.concat([PCAdataframe,data[['Whole_Weight']]],axis=1)
    print(ConcatComponents)
    file5.write("\n%s" %ConcatComponents)
    print("\nPCA Explained Variance: %s" %pca.explained_variance_ratio_)
    file5.write("\nPCA Explained Variance: %s" %pca.explained_variance_ratio_)
    print("\nPCA Components: %s" %pca.components_)
    file5.write("\nPCA Components: %s" %pca.components_)
    scalar=StandardScaler()
    scalar.fit(x_train)
    x_train=scalar.transform(x_train)
    x_test=scalar.transform(x_test)
    pca=PCA(n_components=K)
    pca.fit(x_train)
    x_train=pca.transform(x_train)
    x_test=pca.transform(x_test)
    model=LinearRegression()
#    y_train=y_train.astype("float64")
#    y_test=y_test.astype("float64")
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    print("\nPrediction After PCA: %s" %prediction)
    file5.write("\nPrediction After PCA: %s" %prediction)
    score=r2_score(y_test,prediction)
    print("\nR^2 After PCA: %s" %score)
    file5.write("\nR^2 After PCA: %s" %score)
    return score
    
def RecFeatureElimination(x_train,y_train,labels):
    
    model=LinearRegression()
    rfe=RFE(model,4)  
    rfe=rfe.fit(x_train,y_train)
    
    boolean=rfe.support_
    print("\nRecursive Feature Elimination Ranking: %s" %rfe.ranking_)
    file4.write("\nRecursive Feature Elimination Ranking: %s" %rfe.ranking_)
    rank=[0,0,0,0]
    count=0
    for i in range(0,len(boolean)):
        if boolean[i] == True:
            rank[count]=i
            temp=labels[i]
            labels[i]=labels[count]
            labels[count]=temp
            count+=1
    print("\nSelected Features are: ")
    file4.write("\nSelected Features are: ")
    for i in range(0,4):
        print(labels[i])
        file4.write("\n %s" %labels[i])
    Selected=X[:,[rank[0],rank[1],rank[2],rank[3]]]
    print("\nData of Selected Features: \n%s" %Selected)
    file4.write("\nData of Selected Features: \n%s" %Selected)
    x_train,x_test,y_train,y_test=train_test_split(Selected,Y,test_size=0.8,random_state=0)
#   y_train=y_train.astype("float64")
#   y_test=y_test.astype("float64")
    model=LinearRegression()
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    print("\nPrediction After RFE: \n%s" %prediction)
    file4.write("\nPrediction After RFE: \n%s" %prediction)
    score=r2_score(y_test,prediction)
    print("\nR^2 After RFE: %s" %score)
    file4.write("\nR^2 After RFE: %s" %score)
    return Selected,score
    

###############################################################################

#Reading Data    
data=pd.read_csv("abalone.csv")
labels=["Sex","Length","Diameter","Height","Whole_Weight","Shucked_Weight","Viscera_Weight","Shell_Weight","Rings"] 
values=data.values
#Y=values[:,4]
#X=values[:,np.r_[0:4,5:9]]     #Extracting multiple columns with multiple ranges
Y=values[:,8]
X=values[:,0:8]
#Encoding categorical data
LE=LabelEncoder()   
X[:,0]=LE.fit_transform(X[:,0])
OHE=OneHotEncoder(categorical_features=[0])
X=OHE.fit_transform(X).toarray()
X = X[:, 1:] 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.67,random_state=0)
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print("\n\n\t\tSimple Prediction Using Linear Regression\n %s" %Y_pred)
file1.write("\n\n\t\tSimple Prediction Using Linear Regression\n %s" %Y_pred)
score=regressor.score(X_test,Y_test)
print("\nR^2: %s" %r2_score(Y_test,Y_pred))
file1.write("\nR^2: %s" %r2_score(Y_test,Y_pred))

###############################################################################
############################  Forward Selection  ##############################
###############################################################################

print("\n\n\t\t\tFORWARD SELECTION\n")
file2.write("\n\n\t\t\tFORWARD SELECTION\n")
scoreFS=Forward_Selection(data,'Rings',labels,Y)

###############################################################################
###############################  Linear SVR  ##################################
###############################################################################

print("\n\n\t\t\tLINEAR SVR\n")
file3.write("\n\n\t\t\tLINEAR SVR\n")
scoreLSVR=LSVR(X_train,X_test,Y_train,Y_test)

###############################################################################
######################  Recursive Feature Elimination  ########################
###############################################################################

print("\n\n\t\tRECURSIVE FEATURE ELIMINATION\n")
file4.write("\n\n\t\tRECURSIVE FEATURE ELIMINATION\n")
#y_train=Y_train.astype(int)
Selected,scoreRE=RecFeatureElimination(X_train,Y_train,labels)

###############################################################################
######################  Principle Component Analysis  #########################
###############################################################################

print("\n\n\t\tPRINCIPLE COMPONENT ANALYSIS\n")
file5.write("\n\n\t\tPRINCIPLE COMPONENT ANALYSIS\n")
scorePCA=PrincipleComponents(4,X,X_train,X_test,Y_train,Y_test)


s=pd.Series(
    [scoreFS,scoreLSVR,scoreRE,scorePCA],
    index=["Forward Selection","LSVR","RFE","PCA"]
)

#Set descriptions:
objects=("Forward Selection","LSVR","RFE","PCA")
plt.ylabel('R^2 Score')
plt.title('Variable Selection')

y_pos=np.arange(len(objects))
plt.xticks(y_pos,objects)
performance=[scoreFS,scoreLSVR,scoreRE,scorePCA]

#Plot the data:
plt.bar(y_pos,performance,align='center',alpha=1,color='rgby',)
ax=plt.gca()
ax.set_ylim([0,0.6])
plt.savefig('Variable Selection Comparision Graph.png',bbox_inches='tight') 
plt.show() 