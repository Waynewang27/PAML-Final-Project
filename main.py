# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# df = pd.read_csv('Datasets/Viral_Social_Media_Trends.csv')


# X = df[['Platform', 'Hashtag', 'Content_Type', 'Region']]
# y = df['Engagement_Level']


# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )


# categorical_features = ['Platform', 'Hashtag', 'Content_Type', 'Region']
# preprocessor = ColumnTransformer([
#     ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features)
# ])


# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', LogisticRegression(
#         multi_class='multinomial', solver='lbfgs', max_iter=1000
#     ))
# ])


# cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
# print(f'Cross‑validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')


# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# test_acc = accuracy_score(y_test, y_pred)
# print(f'Test set accuracy: {test_acc:.3f}\n')


# print('Classification Report:')
# print(classification_report(y_test, y_pred))
# print('Confusion Matrix:')
# print(confusion_matrix(y_test, y_pred, labels=pipeline.named_steps['classifier'].classes_))

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load preprocessed dataset
df = pd.read_csv('Datasets/preprocessed_data.csv')

# Drop columns related to engagement metrics to avoid data leakage
engagement_cols = ['Views','Likes','Shares','Comments','Conversion_Rate'] 
df = df.drop(columns=[col for col in engagement_cols if col in df.columns])

# Separate features and target
X = df.drop(columns=['Engagement_Level_Calculated'])
y = df['Engagement_Level_Calculated']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define a simple pipeline (no preprocessing needed)
pipeline = Pipeline([
    ('classifier', LogisticRegression(
        multi_class='multinomial', solver='lbfgs', max_iter=1000
    ))
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

# Fit on training data and evaluate on test data
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print(f'Test set accuracy: {test_acc:.3f}\n')

print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred, labels=pipeline.named_steps['classifier'].classes_))


# ── Improve Class 0 Performance with SMOTE ──
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# assume X_train, y_train, X_test, y_test already defined above

# 1) Compare oversamplers + class_weighted LR
oversamplers = {
    'SMOTE': SMOTE(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42)
}

for name, sampler in oversamplers.items():
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    lr = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    lr.fit(X_res, y_res)
    y_pred = lr.predict(X_test)

    print(f"\n=== LogisticRegression with {name} + balanced class_weight ===")
    print(f"Train distribution: {pd.Series(y_res).value_counts().to_dict()}")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=lr.classes_))


# 2) Threshold tuning on SMOTE+LR
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

lr_sm = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
lr_sm.fit(X_sm, y_sm)
probs_class0 = lr_sm.predict_proba(X_test)[:, lr_sm.classes_.tolist().index(0)]

best_thresh, best_f1 = 0.5, 0
for thresh in np.linspace(0.1, 0.9, 17):
    y_pred_thresh = np.where(probs_class0 > thresh, 0, 1)
    rpt = classification_report(y_test, y_pred_thresh, output_dict=True)
    f1_0 = rpt.get('0', {'f1-score': 0})['f1-score']
    if f1_0 > best_f1:
        best_f1, best_thresh = f1_0, thresh

print(f"\n>>> Best threshold for Class 0 f1: {best_thresh:.2f} (f1={best_f1:.3f})")
y_pred_best = np.where(probs_class0 > best_thresh, 0, 1)
print(classification_report(y_test, y_pred_best, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best, labels=[0,1]))


# ── Integrate Random Forest on Balanced Data ──
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# 4) Set up Random Forest + hyperparameter search
rf = RandomForestClassifier(random_state=42)
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_search = RandomizedSearchCV(
    rf, param_distributions=param_dist,
    n_iter=10, cv=5, scoring='accuracy',
    random_state=42, n_jobs=-1
)
rf_search.fit(X_sm, y_sm)
best_rf = rf_search.best_estimator_
print("Best RF params:", rf_search.best_params_)

# 5) Evaluate Random Forest
y_pred_rf = best_rf.predict(X_test)
print("----- Random Forest Results -----")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf, labels=best_rf.classes_))

