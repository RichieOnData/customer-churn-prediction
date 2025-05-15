import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load preprocessed data
df = pd.read_csv('processed_customer_churn.csv')

# 2. Feature/target split
X = df.drop(['CustomerID', 'ChurnStatus'], axis=1)
y = df['ChurnStatus']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model selection rationale
print("""
Selected Algorithm: Random Forest Classifier
Rationale: Random Forests provide a strong balance between accuracy and interpretability. They handle feature interactions, are robust to outliers, and provide feature importance scores, which are valuable for business insights. They also perform well on tabular data and can handle imbalanced classes with class weighting.
""")

# 5. Hyperparameter tuning with cross-validation
grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
rf = RandomForestClassifier(random_state=42)
gs = GridSearchCV(rf, grid, cv=5, scoring='f1', n_jobs=-1)
gs.fit(X_train, y_train)

print(f"Best parameters: {gs.best_params_}")

# 6. Evaluate on test set
best_rf = gs.best_estimator_
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc:.3f}")

# 7. Confusion matrix plot
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# 8. ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# 9. Feature importance plot
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# 10. Business recommendations
print("""
Business Recommendations:
- Use the model to identify customers with high predicted churn probability and target them with retention offers.
- Focus on the most important features (see 'feature_importances.png') for actionable insights (e.g., high unresolved service issues, low login frequency).
- Regularly retrain the model with new data to maintain accuracy.
- Consider further improvements: try other algorithms (e.g., XGBoost), advanced feature engineering, or ensemble methods.
""") 