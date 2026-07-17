#IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#LOAD DATASET
print("\n------ LOAD DATA ------\n")
df=pd.read_csv(r"C:\Users\user\Downloads\Data Train.csv")
print(df.head())

#UNDERSTAND THE DATA
print("\n------ DATA INFORMATION ------\n")
print("Shape of Dataset:")
print(df.shape)
print("\nColumn Names:")
print(df.columns)
print("\nInformation:")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

#DATA PREPROCESSING
print("\n------ DATA PREPROCESSING ------\n")

#Fill Missing Values
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=df[col].fillna(df[col].mode()[0])
    else:
        df[col]=df[col].fillna(df[col].median())

print(df.head())

#LABEL ENCODING
print("\n------ LABEL ENCODING ------\n")
encoder=LabelEncoder()
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=encoder.fit_transform(df[col])
print(df.head())

#EXPLORATORY DATA ANALYSIS
print("\n------ EDA ------\n")
print(df["churn"].value_counts())

#Churn Count
plt.figure(figsize=(5,4))
df["churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

#Account Length Histogram
plt.figure(figsize=(5,5))
plt.hist(df["account_length"],bins=20)
plt.title("Account Length Distribution")
plt.xlabel("Account Length")
plt.ylabel("Frequency")
plt.show()

#Total Day Minutes Histogram
plt.figure(figsize=(5,5))
plt.hist(df["total_day_minutes"],bins=20)
plt.title("Total Day Minutes Distribution")
plt.xlabel("Total Day Minutes")
plt.ylabel("Frequency")
plt.show()

#Correlation Matrix
plt.figure(figsize=(10,8))
plt.imshow(df.corr(),cmap="YlGnBu")
plt.colorbar()
plt.xticks(range(len(df.columns)),df.columns,rotation=90)
plt.yticks(range(len(df.columns)),df.columns)
plt.title("Correlation Matrix")
plt.show()

#FEATURE SCALING
print("\n------ FEATURE SCALING ------\n")
X=df.drop("churn",axis=1)
y=df["churn"]
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print("Scaled Features:")
print(X_scaled[:5])

#TRAIN TEST SPLIT
print("\n------ TRAIN TEST SPLIT ------\n")
X_train,X_test,y_train,y_test=train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Samples:",len(X_train))
print("Testing Samples:",len(X_test))

#MODEL
print("\n------ TRAINING MODEL ------\n")
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
print("Model Training Completed.")

#PREDICTION
print("\n------ TEST MODEL ------\n")
y_pred=model.predict(X_test)
print("Predicted Values:")
print(y_pred)

#MODEL EVALUATION
print("\n------ MODEL EVALUATION ------\n")
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#NEW CUSTOMER PREDICTION
print("\n------ NEW PREDICTION ------\n")
new_customer = [[
    16,      #state
    128,     #account_length
    1,       #area_code
    0,       #international_plan
    1,       #voice_mail_plan
    25,      #number_vmail_messages
    265.1,   #total_day_minutes
    110,     #total_day_calls
    45.07,   #total_day_charge
    197.4,   #total_eve_minutes
    99,      #total_eve_calls
    16.78,   #total_eve_charge
    244.7,   #total_night_minutes
    91,      #total_night_calls
    11.01,   #total_night_charge
    10.0,    #total_intl_minutes
    3,       #total_intl_calls
    2.70,    #total_intl_charge
    1        #number_customer_service_calls
]]
new_scaled=scaler.transform(new_customer)
prediction=model.predict(new_scaled)
if prediction[0]==1:
    print("Customer will churn.")
else:
    print("Customer will not churn.")

#Accuracy
print("\nFinal Accuracy:",accuracy)

