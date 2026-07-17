#IMPORT LIBRARIES
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

#HANDLE MISSING VALUES
for col in df.columns:
    if df[col].dtype=="object":
        df[col]=df[col].fillna(df[col].mode()[0])
    else:
        df[col]=df[col].fillna(df[col].median())

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

#RANDOM FOREST MODEL
print("\n------ RANDOM FOREST ------\n")
model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

#PREDICTION
print("\n------ TEST MODEL ------\n")
y_pred=model.predict(X_test)
print("Predicted Values:")
print(y_pred)

#MODEL EVALUATION
print("\n------ MODEL EVALUATION ------\n")
accuracy=accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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

#Values must be in the same order as X.columns
new_customer = [[
    16,      #state (encoded value)
    128,     #account_length
    1,       #area_code (encoded)
    0,       #international_plan (0=No,1=Yes)
    1,       #voice_mail_plan (0=No,1=Yes)
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

prediction=model.predict(new_customer)

if prediction[0]==1:
    print("Customer is likely to Churn.")
else:
    print("Customer is likely to Stay.")

#FINAL ACCURACY
print("\nFinal Accuracy:",accuracy)
