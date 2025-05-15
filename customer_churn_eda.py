import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Set style for better visualizations
plt.style.use('default')
sns.set_theme()

def load_and_merge_data():
    """Load all relevant sheets and merge them on CustomerID"""
    print("Loading data from all sheets...")
    xls = pd.ExcelFile('Customer_Churn_Data_Large.xlsx')
    transaction = pd.read_excel(xls, 'Transaction_History')
    service = pd.read_excel(xls, 'Customer_Service')
    online = pd.read_excel(xls, 'Online_Activity')
    churn = pd.read_excel(xls, 'Churn_Status')

    # Aggregate transaction data per customer
    transaction_agg = transaction.groupby('CustomerID').agg({
        'AmountSpent': ['sum', 'mean', 'count'],
        'ProductCategory': pd.Series.nunique
    })
    transaction_agg.columns = ['TotalSpent', 'AvgSpent', 'NumTransactions', 'NumProductCategories']
    transaction_agg.reset_index(inplace=True)

    # Aggregate customer service data per customer
    service_agg = service.groupby('CustomerID').agg({
        'InteractionID': 'count',
        'InteractionType': pd.Series.nunique,
        'ResolutionStatus': lambda x: (x == 'Unresolved').sum()
    })
    service_agg.columns = ['NumInteractions', 'NumInteractionTypes', 'NumUnresolved']
    service_agg.reset_index(inplace=True)

    # Aggregate online activity per customer
    online_agg = online.groupby('CustomerID').agg({
        'LoginFrequency': 'mean',
        'ServiceUsage': pd.Series.nunique
    })
    online_agg.columns = ['AvgLoginFrequency', 'NumServiceUsageTypes']
    online_agg.reset_index(inplace=True)

    # Merge all features
    df = churn.merge(transaction_agg, on='CustomerID', how='left')
    df = df.merge(service_agg, on='CustomerID', how='left')
    df = df.merge(online_agg, on='CustomerID', how='left')

    print("\nMerged Data Info:")
    print(df.info())
    print(df.head())
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("\nMissing Values Analysis:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Fill missing values with appropriate methods
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].median(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)
    
    return df

def analyze_distributions(df):
    """Analyze and visualize distributions of numerical features"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['CustomerID', 'ChurnStatus'])
    
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue='ChurnStatus', multiple="stack")
        plt.title(f'Distribution of {col} by Churn Status')
        plt.savefig(f'plots/distribution_{col}.png')
        plt.close()

def analyze_categorical_features(df):
    """Analyze categorical features"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=col, hue='ChurnStatus')
        plt.title(f'{col} Distribution by Churn Status')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'plots/categorical_{col}.png')
        plt.close()

def correlation_analysis(df):
    """Perform correlation analysis"""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop(['CustomerID', 'ChurnStatus'])
    
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

def preprocess_data(df):
    """Preprocess the data for modeling"""
    processed_df = df.copy()
    
    # Handle categorical variables
    categorical_cols = processed_df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        processed_df = pd.get_dummies(processed_df, columns=categorical_cols, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = processed_df.select_dtypes(include=['int64', 'float64']).columns.drop(['CustomerID', 'ChurnStatus'])
    processed_df[numerical_cols] = scaler.fit_transform(processed_df[numerical_cols])
    
    return processed_df

def main():
    # Load and explore data
    df = load_and_merge_data()
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Analyze distributions
    analyze_distributions(df)
    
    # Analyze categorical features
    analyze_categorical_features(df)
    
    # Perform correlation analysis
    correlation_analysis(df)
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    # Save processed data
    processed_df.to_csv('processed_customer_churn.csv', index=False)
    print("\nProcessed data saved to 'processed_customer_churn.csv'")
    
    print("\nEDA completed successfully! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main() 