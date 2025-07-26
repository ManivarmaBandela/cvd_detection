# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Step 1: Loading the Dataset
print("Step 1: Loading the Dataset")
try:
    df = pd.read_csv('heart_disease_data.csv')  # Update with your dataset path
    print("Initial Data Shape:", df.shape)
except FileNotFoundError:
    print("Dataset file not found. Using sample data for demonstration.")
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 300
    age = np.random.randint(30, 80, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)  # chest pain type
    trestbps = np.random.randint(90, 200, n_samples)  # resting blood pressure
    chol = np.random.randint(120, 400, n_samples)  # cholesterol
    fbs = np.random.randint(0, 2, n_samples)  # fasting blood sugar
    restecg = np.random.randint(0, 3, n_samples)  # resting ECG
    thalach = np.random.randint(70, 210, n_samples)  # max heart rate
    exang = np.random.randint(0, 2, n_samples)  # exercise induced angina
    oldpeak = np.random.uniform(0, 6, n_samples)  # ST depression
    slope = np.random.randint(0, 3, n_samples)  # slope of peak exercise ST
    ca = np.random.randint(0, 4, n_samples)  # number of major vessels
    thal = np.random.randint(0, 3, n_samples)  # thalassemia
    
    # Create target variable with some correlation to features
    target = (age > 55) & (chol > 250) | (trestbps > 140) & (exang == 1)
    target = target.astype(int)
    
    # Create DataFrame
    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope,
        'ca': ca, 'thal': thal, 'target': target
    }
    df = pd.DataFrame(data)
    print("Sample Data Shape:", df.shape)

# Step 2: Data Exploration
print("\nStep 2: Data Exploration")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values)

# Step 3: Handling Missing Values
print("\nStep 3: Handling Missing Values")
if missing_values.sum() > 0:
    print("Filling missing values with median for numeric columns")
    df.fillna(df.median(), inplace=True)
    print("Missing values after filling:", df.isnull().sum().sum())
else:
    print("No missing values found in the dataset")

# Step 4: Feature Engineering
print("\nStep 4: Feature Engineering")
print("For this example, we'll use the existing features without transformation")

# Step 5: Preparing the Data for Model Training
print("\nStep 5: Preparing the Data for Model Training")
# Define features and target variable
X = df.drop('target', axis=1)  # Assuming 'target' is the column to predict
y = df['target']

print(f"Features shape: {X.shape}, Target shape: {y.shape}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed")

# Step 6: Model Development
print("\nStep 6: Model Development")
# Initialize models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary to store results
results = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    print(f"{model_name} trained successfully")

# Step 7: Model Evaluation
print("\nStep 7: Model Evaluation")
results_df = pd.DataFrame(results).T
print("\nModel Evaluation Metrics:")
print(results_df)

# Identify the best performing model based on F1 Score
best_model = results_df['F1 Score'].idxmax()
print(f"\nBest Performing Model: {best_model} with F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f}")

# Step 8: Insights and Reporting
print("\nStep 8: Insights and Reporting")
# Feature Importance using the best model
if best_model == 'Random Forest' or best_model == 'Decision Tree':
    importance_model = models[best_model]
    feature_importances = importance_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    print(feature_importance_df.head(5))
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title(f'Feature Importance for {best_model}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Feature importance plot saved as 'feature_importance.png'")

# Detailed classification report for the best model
print("\nClassification Report for the Best Model:")
best_model_pred = models[best_model].predict(X_test_scaled)
print(classification_report(y_test, best_model_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = pd.crosstab(y_test, best_model_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for {best_model}')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix plot saved as 'confusion_matrix.png'")

print("\nCardiovascular Disease Detection Analysis Complete!")