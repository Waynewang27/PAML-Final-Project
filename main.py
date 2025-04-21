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
