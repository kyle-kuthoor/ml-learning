import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
print("Libraries loaded")
# Load the dataset
df = pd.read_csv("/Users/kylesteny/social-media-ad-analysis/Social_Media_Advertising.csv")

# We'll predict ROI based on these features
# First let's look at what we're working with
print("Shape:", df.shape)
print("\nFeatures we'll use:")
print(df[['Clicks', 'Impressions', 'Engagement_Score', 'Conversion_Rate']].describe())
print("\nTarget variable (what we're predicting):")
print(df['ROI'].describe())
# Define features and target
X = df[['Clicks', 'Impressions', 'Engagement_Score', 'Conversion_Rate']]
y = df['ROI']

# Split into training and testing sets
# test_size=0.2 means 20% of data is kept for testing, 80% for training
# random_state=42 makes the split consistent every time we run it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train):,} rows")
print(f"Testing set size:  {len(X_test):,} rows")
# Create the model
model = LinearRegression()

# Train it — this is where the model learns from the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate how good the predictions are
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"  R² Score:            {r2:.4f}")
print(f"  Mean Squared Error:  {mse:.4f}")
print(f"\nSample predictions vs actual:")
for actual, predicted in list(zip(y_test[:5], y_pred[:5])):
    print(f"  Actual: {actual:.2f}  |  Predicted: {predicted:.2f}")
# One-hot encode the Channel_Used column
# This turns "Instagram", "Facebook" etc into separate 0/1 columns
df_encoded = pd.get_dummies(df, columns=['Channel_Used'], drop_first=True)

print("New columns after encoding:")
new_cols = [c for c in df_encoded.columns if 'Channel' in c]
print(new_cols)
# New feature set including platform columns
X2 = df_encoded[['Clicks', 'Impressions', 'Engagement_Score', 'Conversion_Rate',
                  'Channel_Used_Instagram', 'Channel_Used_Pinterest', 'Channel_Used_Twitter']]
y2 = df_encoded['ROI']

# Split again
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Train new model
model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

r2_new = r2_score(y2_test, y2_pred)
mse_new = mean_squared_error(y2_test, y2_pred)

print(f"Model 1 (without platform):  R² = 0.2462")
print(f"Model 2 (with platform):     R² = {r2_new:.4f}")
print(f"\nImprovement: {((r2_new - 0.2462) / 0.2462 * 100):.1f}%")

# Show feature importance
print(f"\nFeature coefficients (impact on ROI):")
for feature, coef in zip(X2.columns, model2.coef_):
    print(f"  {feature:<35} {coef:+.4f}")
# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y2_test[:1000], y2_pred[:1000], alpha=0.3, color='#4a7c3f', s=10)
plt.plot([0, 8], [0, 8], color='red', linestyle='--', label='Perfect prediction')
plt.xlabel('Actual ROI')
plt.ylabel('Predicted ROI')
plt.title('Actual vs Predicted ROI — Linear Regression Model', fontsize=13)
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.close()
print("Chart saved.")