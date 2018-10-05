#%%Group Project
#%% Credit Lending
#%% import libraries

#%%
import pandas as pd
import numpy as np

#%% read file
network_data = pd.read_csv(r'../input/XYZCorp_LendingData1.txt',sep=' ', delimiter='\t')
#%%  check shape
print(network_data.head())
#%%Displaying Max rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#%%Describe
network_data.describe (include='all')
#%%Checking Missing Values
print(network_data.isnull().sum()) 
#%% making copy my_df
my_df=pd.DataFrame.copy(network_data)
#%%Checking Missing Values
print(my_df.isnull().sum())

#%% graph
sns.countplot(x = 'verification_status', data = my_df)
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
#%%Checking for data which have more than 50% missing values
my_df.isnull().sum()/855969*100
#%%Dropping Columns unwanted columns 
my_df_rev=my_df.drop(['emp_title','title','last_pymnt_d','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med','tot_coll_amt','id','member_id','pymnt_plan','zip_code','addr_state','earliest_cr_line','initial_list_status','last_pymnt_amnt','policy_code','recoveries','total_rec_int','total_rev_hi_lim','total_rec_late_fee','collection_recovery_fee', 'out_prncp', 'funded_amnt_inv', 'grade', 'verification_status', 'out_prncp_inv', 'total_pymnt','total_pymnt_inv','total_rec_prncp','application_type'], axis=1)
#%%Checking Missing Value
print(my_df_rev.isnull().sum())
#%%  fill missing values with mean values for Total Current Balance 
my_df_rev['tot_cur_bal'].fillna(my_df_rev['tot_cur_bal'].mean(), inplace=True)
#%%  fill missing values with mean values for revol_util
my_df_rev['revol_util'].fillna(my_df_rev['revol_util'].mean(), inplace=True)
#%%Checking Missing Values
print(my_df_rev.isnull().sum())

#%% Converting issue_d to to_datetime
my_df_rev['issue_d']=pd.to_datetime(my_df_rev.issue_d)
#%% Creating ts to store midpoint
ts=pd.to_datetime('May-2015')
#%% Splitting Data in train and test using ts as midpoint reference
train_data=my_df_rev.loc[my_df_rev.issue_d<=ts, :]
#%% Splitting Data in train and test using ts as midpoint reference
test_data=my_df_rev.loc[my_df_rev.issue_d>ts, :]
#%%Checking if the split is correct
train_data['issue_d'].value_counts().sort_index()
#%%Checking if the split is correct
test_data['issue_d'].value_counts().sort_index()
#%%Checking Data types
print(train_data.dtypes)
#print(test_data.dtypes)
#%% Dropping the issue_d column
train_data=train_data.drop(['issue_d',], axis=1)
#%% Dropping the issue_d column
test_data=test_data.drop(['issue_d'], axis=1)
#%% for preprocessing the data - creating categorical data in numerical form
from sklearn import preprocessing
colname=['grade', 'home_ownership','application_type','term','verification_status']

le={}
    
for x in colname:
        le[x]=preprocessing.LabelEncoder()
for x in colname:
        train_data[x]=le[x].fit_transform(train_data.__getattr__(x))
    
#%% for preprocessing the data - creating categorical data in numerical form
from sklearn import preprocessing
colname=[ 'grade', 'home_ownership','application_type','term','verification_status']

le={}
    
for x in colname:
        le[x]=preprocessing.LabelEncoder()
for x in colname:
        test_data[x]=le[x].fit_transform(test_data.__getattr__(x))
    

#%%    create x and y     #all variables except last one
        
     x_train=train_data.values[:,:-1] 
     y_train=train_data.values[:,-1] #subset rows , subset columns 
     y_train=y_train.astype(int) #sometimes y is treated as object so typecast y as int
     #%% scale data
     x_test=test_data.values[:,:-1]
     y_test=test_data.values[:,-1] #subset rows , subset columns 
     y_test=y_test.astype(int) #sometimes y is treated as object so typecast y as int
#%% scale data train
     from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    print(x_train)
   
#%% scale data test
     from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test=scaler.transform(x_test)
    print(x_test)
  

#%%running the model
from sklearn.linear_model import LogisticRegression
#create a model
classifier=(LogisticRegression()) 
 #fitting training data to the model
classifier.fit(x_train,y_train) 
y_pred=classifier.predict(x_test)
print(list(zip(y_test, y_pred)))
#print(y_test)

 
#%% checking accuracy
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test,y_pred)
print(cfm) 
print("Classification report:" )
print(classification_report(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy of the model: " , accuracy_score)
LogisticRegression_acc=accuracy_score     
#%%adjusting the threshold
#storing the predicted probabilities

y_pred_prob=classifier.predict_proba(x_test)
print(y_pred_prob)

#%%
y_pred_class=[]
for value in y_pred_prob[:,0]:
    if value<0.75:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
#%% checking accuracy
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cfm=confusion_matrix(y_test.tolist(),y_pred_class)
print(cfm) 
print(classification_report(y_test.tolist(),y_pred_class))

accuracy_score=accuracy_score(y_test.tolist(),y_pred_class)
print("Accuracy of the model: " , accuracy_score)
AfteradjThreshold_acc=accuracy_score


#%%.................................................
#%%Predicting using the Decision Tree classifier
 from sklearn.tree import DecisionTreeClassifier
#create a model
 
 model_DecisionTree=DecisionTreeClassifier()
 
 #fitting training data to the model
model_DecisionTree.fit(x_train,y_train) 
y_pred=model_DecisionTree.predict(x_test)
#print(list(zip(y_test, y_pred)))
#print(y_pred)


#%% checking accuracy
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
 
print(confusion_matrix(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy of the model: " , accuracy_score)
print("Classification report:" )
print(classification_report(y_test,y_pred))
DecisionTree_acc=accuracy_score

#%% Running extra trees classifier model
#Predicting using the Bagging_Classifier

from sklearn.ensemble import ExtraTreesClassifier
model=(ExtraTreesClassifier(21))
#fit the model on the data and predict the values
model=model.fit(x_train, y_train) #training the model
y_pred=model.predict(x_test)
#%% check accuracy - confusion matrix 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
 
print(confusion_matrix(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy of the model: " , accuracy_score)
print("Classification report:" )
print(classification_report(y_test,y_pred))

Bagging_acc=accuracy_score
#%%Running Adaboost classifier
#Predicting using the Adaboost classifier

from sklearn.ensemble import AdaBoostClassifier
model_AdaBoost=(AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))
#fit the mobel on the data and predict the values
model_AdaBoost.fit(x_train, y_train) 
y_pred=model_AdaBoost.predict(x_test)


#%% check accuracy - confusion matrix again
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
 
print(confusion_matrix(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy of the model: " , accuracy_score)
print("Classification report:" )
print(classification_report(y_test,y_pred))
Adaboost_acc=accuracy_score
#%%Running Gradient Boosting classifier
#Predicting using the GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
model_GradientBoosting=GradientBoostingClassifier()
#fit the mobel on the data and predict the values
model_GradientBoosting.fit(x_train, y_train) 
y_pred=model_GradientBoosting.predict(x_test)


#%% check accuracy 
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
 
print(confusion_matrix(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy of the model: " , accuracy_score)
print("Classification report:" )
print(classification_report(y_test,y_pred))

GradientBoosting_acc=accuracy_score
#%% Running Random Forest model
#Predicting using the RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier
model_RandomForest=RandomForestClassifier(501)
#fit the model on the data and predict the values
model_RandomForest.fit(x_train, y_train) #training the model
y_pred=model.RandomForest.predict(x_test)


#%% check accuracy - confusion matrix again

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
 
print(confusion_matrix(y_test,y_pred))
accuracy_score=accuracy_score(y_test,y_pred)
print("Accuracy of the model: " , accuracy_score)
print("Classification report:" )
print(classification_report(y_test,y_pred))

RandomForest_acc=accuracy_score
#%%

print("Logistic Regression:",LogisticRegression_acc*100,'%',"\n After Adj Threshold:",AfteradjThreshold_acc*100,'%',"\n Decision Tree:",DecisionTree_acc*100,'%',"\ Bagging:",Bagging_acc*100,'%',
      "\n Adaboost:",Adaboost_acc*100,'%',"\n Gradient Boosting:",GradientBoosting_acc*100,'%', "\n RandomForest_acc:",RandomForest_acc*100,'%',")

