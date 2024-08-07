import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataset = pd.read_csv('./Placement_Data_Full_Class.csv')
print(dataset.head())
dataset.drop(['salary','sl_no'], axis=1, inplace=True)
dataset.isnull().sum()
features_to_split = ['hsc_s','degree_t']
for feature in features_to_split:
    dummy = pd.get_dummies(dataset[feature])
    dataset = pd.concat([dataset, dummy], axis=1)
    dataset.drop(feature, axis=1, inplace=True)
dataset.rename(columns={"Others": "Other_Degree"},inplace=True)    

encoder = LabelEncoder()
columns_to_encode = ['gender','ssc_b', 'hsc_b','workex','specialisation','status']
for column in columns_to_encode:
    dataset[column] = encoder.fit_transform(dataset[column])
fig, axs = plt.subplots(ncols=6,nrows=3,figsize=(20,10))
index = 0
axs = axs.flatten()
for k,v in dataset.items():
    sns.boxplot(y=v, ax=axs[index])
    index+=1

fig.delaxes(axs[index])
plt.tight_layout(pad=0.3, w_pad=0.5,h_pad = 4.5)     

dataset = dataset[~(dataset['degree_p']>=90)]
dataset = dataset[~(dataset['hsc_p']>=95)]
print(dataset.corr())
x = dataset.loc[:,dataset.columns!='status'] # all features are used
y = dataset.loc[:, 'status'] # label is status of placement
sc= StandardScaler()
x_scaled = sc.fit_transform(x) # for standardising the features
x_scaled = pd.DataFrame(x_scaled)
x_train,x_test, y_train, y_test = train_test_split(x_scaled,y,test_size=0.18, random_state=0)
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
nbclassifier = GaussianNB()
nbclassifier.fit(x_train, y_train)
y_pred_nb = nbclassifier.predict(x_test)
print("Accuracy Score",accuracy_score(y_test, y_pred_nb))
nbclassifier.score(x_train, y_train)
target_names=["NotPlaced","Placed"]
print(f'Classfication Report\n{classification_report(y_test,y_pred_nb,target_names=target_names)}')
print("Confusiion Matrix",confusion_matrix(y_test,y_pred_nb))
print(x_test)
print(y_pred_nb)

for p in y_pred_nb:
    if p==1:
        print("Placed")
    else:
        print("NotPlaced")    