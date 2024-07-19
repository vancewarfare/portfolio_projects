import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')

# Data Cleaning
df.dropna(inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Exploratory Data Analysis (EDA)
# Sales by Category (using StockCode as a proxy for category)
category_sales = df.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)

# Top 10 Products by Sales
top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

# Sales Trend Over Time
monthly_sales = df.set_index('InvoiceDate').resample('M')['Quantity'].sum()

# Visualizations
plt.figure(figsize=(10, 6))
sns.barplot(x=category_sales.index, y=category_sales.values)
plt.title('Top 10 Stock Codes by Quantity Sold')
plt.ylabel('Quantity Sold')
plt.xlabel('Stock Code')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=top_products.index, y=top_products.values)
plt.title('Top 10 Products by Quantity Sold')
plt.ylabel('Quantity Sold')
plt.xlabel('Product')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values)
plt.title('Monthly Sales Trend')
plt.ylabel('Quantity Sold')
plt.xlabel('Date')
plt.grid(True)
plt.show()
