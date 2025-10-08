#impoprting libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

#preparing data
column_names = []
with open("spambase/spambase.names") as f:
    for line in f:
        if line.strip() and ":" in line:
            col_name = line.split(":")[0].strip()
            column_names.append(col_name)
column_names.append("is_spam")
df = pd.read_csv("spambase/spambase.data" , names=column_names)


#working on model
X = df.drop("is_spam", axis=1).values
y= df["is_spam"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg = LogisticRegression(max_iter=1000,solver='lbfgs')

logreg.fit(X_train_scaled,y_train)

y_pred = logreg.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))