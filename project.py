
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore


# -----------------------------
np.random.seed(42)
n_samples = 200
ages = np.random.randint(60, 90, n_samples)
mmse_scores = np.random.randint(18, 31, n_samples)
cdr_scores = np.round(np.random.choice([0.0, 0.5, 1.0, 2.0], n_samples), 1)
eTIV = np.random.randint(1450, 1610, n_samples)
nWBV = np.round(np.random.uniform(0.65, 0.85, n_samples), 2)
ASF = np.round(np.random.uniform(1.05, 1.35, n_samples), 2)

diagnosis = []
for i in range(n_samples):
    prob = 0.0
    if ages[i] > 75: prob += 0.3
    if mmse_scores[i] < 25: prob += 0.5
    if cdr_scores[i] >= 1.0: prob += 0.2
    diagnosis.append(1 if np.random.rand() < prob else 0)

df = pd.DataFrame({
    'Age': ages,
    'MMSE': mmse_scores,
    'CDR': cdr_scores,
    'eTIV': eTIV,
    'nWBV': nWBV,
    'ASF': ASF,
    'Diagnosis': diagnosis
})


# -----------------------------
X = df[['Age', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# -----------------------------
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)


# -----------------------------
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}")
print(classification_report(y_test, y_pred))


# -----------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values('Coefficient', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette='coolwarm')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.tight_layout()
plt.show()


# -----------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Demented','Demented'], yticklabels=['Non-Demented','Demented'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# -----------------------------
def predict_alzheimers(age, mmse, cdr, etiv, nwbv, asf):
    features = np.array([[age, mmse, cdr, etiv, nwbv, asf]])
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0][1]
    return probability * 100  # percentage


prob = predict_alzheimers(78, 23, 1.0, 1500, 0.71, 1.26)
print(f"\nPredicted Alzheimer's Probability: {prob:.2f}%")




