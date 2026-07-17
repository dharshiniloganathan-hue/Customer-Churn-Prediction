#IMPORT LIBRARIES
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#LOAD DATASET
print("\n------ LOAD DATA ------\n")
df = pd.read_csv(r"C:\Users\user\Downloads\Data Train.csv")
print(df.head())

#UNDERSTAND THE DATA
print("\n------ DATA INFORMATION ------\n")
print("Shape:",df.shape)
print("\nColumns:")
print(df.columns)
print("\nInformation:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

#DROP UNNECESSARY COLUMN
if "phone_number" in df.columns:
    df=df.drop("phone_number",axis=1)

#LABEL ENCODING
print("\n------ LABEL ENCODING ------\n")
encoder=LabelEncoder()
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=encoder.fit_transform(df[col])
print(df.head())

#FEATURES AND TARGET
X=df.drop("churn", axis=1)
y=df["churn"]

#TRAIN TEST SPLIT
print("\n------ TRAIN TEST SPLIT ------\n")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
print("Training Samples:",len(X_train))
print("Testing Samples:",len(X_test))

#DECISION TREE MODEL
print("\n------ DECISION TREE MODEL ------\n")
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)

#PREDICTION
print("\n------ TEST MODEL ------\n")
y_pred=model.predict(X_test)
print("Predicted Values:")
print(y_pred)

#MODEL EVALUATION
print("\n------ MODEL EVALUATION ------\n")
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

#FEATURE IMPORTANCE
print("\n------ FEATURE IMPORTANCE ------\n")
importance=pd.DataFrame({
    "Feature":X.columns,
    "Importance":model.feature_importances_
})
importance=importance.sort_values(
    by="Importance",
    ascending=False
)
print(importance)

#NEW CUSTOMER PREDICTION
print("\n------ NEW CUSTOMER PREDICTION ------\n")
print("Feature Order:")
print(X.columns)
new_customer=[[
    1,      #state (encoded)
    120,    #account_length
    1,      #area_code (encoded)
    0,      #international_plan (No=0, Yes=1)
    1,      #voice_mail_plan (Yes=1, No=0)
    30,     #number_vmail_messages
    160.5,  #total_day_minutes
    110,    #total_day_calls
    27.29,  #total_day_charge
    180.2,  #total_eve_minutes
    95,     #total_eve_calls
    15.32,  #total_eve_charge
    200.5,  #total_night_minutes
    90,     #total_night_calls
    9.02,   #total_night_charge
    10.5,   #total_intl_minutes
    3,      #total_intl_calls
    2.84,   #total_intl_charge
    1,      #customer_service_calls
]]

prediction=model.predict(new_customer)
if prediction[0]==1:
    print("Customer is likely to Churn.")
else:
    print("Customer is likely to Stay.")

# TEST MODEL
print("\n------ ACCURACY ------\n")
y_pred=model.predict(X_test)
print("Predicted Values:")
print(y_pred)

# TRAINING ACCURACY
train_pred=model.predict(X_train)
train_accuracy=accuracy_score(y_train,train_pred)
print("\nTraining Accuracy:",train_accuracy)

# TESTING ACCURACY
test_accuracy=accuracy_score(y_test,y_pred)
print("\nTesting Accuracy:",test_accuracy)

# OVERALL ACCURACY
overall_pred=model.predict(X)
overall_accuracy=accuracy_score(y,overall_pred)
print("\nOverall Accuracy:",overall_accuracy)
