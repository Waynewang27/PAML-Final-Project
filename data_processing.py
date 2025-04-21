import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Datasets/Viral_Social_Media_Trends.csv")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Check for missing values
missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts)

# Impute missing values if any (mode for categorical, median for numeric)
for col in df.select_dtypes(include='object'):
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ['Views', 'Likes', 'Shares', 'Comments']:
    df[col] = df[col].fillna(df[col].median())

# Cap outliers (top 1% of Views)
cap = df['Views'].quantile(0.99)
df['Views'] = np.where(df['Views'] > cap, cap, df['Views'])

# Logical consistency: Views ≥ Likes + Shares + Comments
df = df[df['Views'] >= (df['Likes'] + df['Shares'] + df['Comments'])]

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['Platform', 'Content_Type', 'Region'])

# Drop non-feature columns
df_encoded = df_encoded.drop(columns=['Post_ID', 'Hashtag'])  # assuming Hashtag won't be modeled directly

# Create 'Conversion Rate' feature
df_encoded['Conversion_Rate'] = df['Shares'] / (df['Likes'] + 1e-5)

# Correlation heatmap for numeric features
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded[['Views', 'Likes', 'Shares', 'Comments',
                        'Conversion_Rate']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()

# Save processed dataset
df_encoded.to_csv("Datasets/processed_social_media_data.csv", index=False)

print("Data processing complete. Processed file saved as 'processed_social_media_data.csv'.")
