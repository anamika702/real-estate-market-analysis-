import pandas as pd
df = pd.read_csv('/var/www/html/data/dataset.csv')
print(df.head())
print(df.info())
print(df.describe())  
print("Missing values per column:\n", df.isnull().sum())
df.dropna(inplace=True)

import seaborn as sns
import matplotlib.pyplot as plt
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

plt.hist(df['MEDV'], bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Median Value of Homes (MEDV)')
plt.ylabel('Frequency')
plt.title('Distribution of Home Prices')
plt.show()

plt.hist(df['RM'], bins=30, color='lightcoral', edgecolor='black')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Rooms')
plt.show()

import statsmodels.api as sm
X = df[['RM', 'LSTAT', 'TAX', 'PTRATIO']] 
y = df['MEDV']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
