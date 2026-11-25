####exploratory data analysis(EDA)-- customer churn analysis
# Import Libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

print("All Libraries Imported Successfully!")

# Load Dataset

df = pd.read_csv("customer_churn.csv")

print("Dataset Loaded Successfully!")


# Check Rows and Columns

df.shape


df.head()

# Display last 5 rows

df.tail()

# Display all column names

df.columns


# Dataset information

df.info()

# Statistical summary

df.describe()


df.describe(include='all')



# Check missing values

df.isnull().sum()


# Total missing values

df.isnull().sum().sum()


# Missing values percentage

missing_percentage = (df.isnull().sum() / len(df)) * 100

missing_percentage




# Visualize missing values

plt.figure(figsize=(12, 6))

sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

plt.title("Missing Values Analysis")
plt.show()



# Check duplicate records

df.duplicated().sum()

# Check duplicate customer IDs

df['customerID'].duplicated().sum()

# Display duplicate records

df[df.duplicated()]

# Remove duplicate records

df= df.drop_duplicates()

print("Duplicate records removed successfully!")



# Check data types

df.dtypes


# Convert TotalCharges into numeric

df['TotalCharges'] = pd.to_numeric(
    df['TotalCharges'].str.strip(),
    errors='coerce'
)

# Verify data types

df['TotalCharges'].dtype

# Check missing values after conversion

df['TotalCharges'].isnull().sum()


# Count unique values in each column

df.nunique()


# Display unique values of all columns

for column in df.columns:
    print(f"\nColumn: {column}")
    print(df[column].unique())


    # Check unique values in Churn

df['Churn'].unique()


# Check unique contract types

df['Contract'].value_counts()


# Gender distribution

df['gender'].value_counts()


# Create a copy of original dataset

df_clean = df.copy()

print("Dataset copy created successfully!")



# Check missing values

df_clean.isnull().sum()

# Check blank spaces in TotalCharges

df_clean['TotalCharges'].astype(str).str.strip().eq('').sum()



# Convert blank spaces into NaN

df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
####df_clean['TotalCharges'] — TotalCharges column को select करता है।
#replace() — पुराने values को नए values से बदलता है।
#r'^\s*$' — ऐसे cells को पहचानता है जिनमें सिर्फ blank spaces हैं।
#np.nan — उन blank spaces को missing value में बदलता है।
#regex=True — बताता है कि pattern को regular expression की तरह समझना है।



# Fill missing values with 0 where tenure is 0

df_clean.loc[
    (df_clean['tenure'] == 0) & (df_clean['TotalCharges'].isnull()),
    'TotalCharges'
] = 0

# df_clean.loc — DataFrame में specific rows और columns select करता है।
# tenure == 0 — ऐसे customers जिनका tenure zero है।
# TotalCharges.isnull() — जिनका TotalCharges missing है।
# & — दोनों conditions एक साथ True होनी चाहिए।
# # 'TotalCharges' — इसी column की value update होगी।
# = 0 — missing value को zero से भर देगा।


# Verify missing values

df_clean['TotalCharges'].isnull().sum()



# Convert TotalCharges into numeric

df_clean['TotalCharges'] = pd.to_numeric(
    df_clean['TotalCharges'],
    errors='coerce'
)



# Check updated data types

df_clean.dtypes




# Check duplicate records

df_clean.duplicated().sum()





# Remove duplicate records

df_clean = df_clean.drop_duplicates()

print("Duplicate records removed successfully!")




# Check duplicate customer IDs

df_clean['customerID'].duplicated().sum()




# Detect outliers using boxplot

plt.figure(figsize=(12, 5))

sns.boxplot(data=df_clean[['tenure', 'MonthlyCharges', 'TotalCharges']])

plt.title("Outlier Detection")

plt.show()
# Outliers वे data points होते हैं जो dataset की बाकी values से काफी अलग होते हैं।
# इसका काम graph का size निर्धारित करना है।
# - plt.figure() — नया graph बनाने के लिए figure तैयार करता है।
# - figsize — graph की width और height निर्धारित करता है।
# - 12 — graph की width 12 inches होगी।
# - 5 — graph की height 5 inches होगी।

# sns — Seaborn library का short name है।
# boxplot() — Boxplot graph बनाता है।
# data= — बताता है कि graph किस dataset से बनाना है।
# df_clean — हमारा cleaned dataset है।
# [['tenure', 'MonthlyCharges', 'TotalCharges']] — इन तीन columns को select करता है।




# Calculate IQR(interQuartile range)

Q1 = df_clean[['tenure', 'MonthlyCharges', 'TotalCharges']].quantile(0.25)

Q3 = df_clean[['tenure', 'MonthlyCharges', 'TotalCharges']].quantile(0.75)

IQR = Q3 - Q1

print("Q1 Values:")
print(Q1)

print("\nQ3 Values:")
print(Q3)

print("\nIQR Values:")
print(IQR)




# Calculate lower and upper limits

lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

print("Lower Limit:")
print(lower_limit)

print("\nUpper Limit:")
print(upper_limit)





# Count outliers in each numerical column

for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:

    outliers = df_clean[
        (df_clean[col] < lower_limit[col]) |
        (df_clean[col] > upper_limit[col])
    ]

    print(f"{col}: {len(outliers)} outliers")





   #### Data Validation


df_clean['gender'].value_counts()

df_clean['Churn'].value_counts()

df_clean['Contract'].value_counts()

df_clean['SeniorCitizen'].value_counts()





# Check negative values in numerical columns

numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

for col in numeric_cols:

    print(col, (df_clean[col] < 0).sum())




# Check final dataset shape

print("Original Dataset Shape:", df.shape)

print("Clean Dataset Shape:", df_clean.shape)




# Check final missing values

df_clean.isnull().sum().sum()




# Check final duplicate records

df_clean.duplicated().sum()



# Display cleaned dataset

df_clean.head()




# Save cleaned dataset

df_clean.to_csv("Customer_Churn_Cleaned.csv", index=False)

print("Clean Dataset Exported Successfully!")



# अगला Step 4: Exploratory Data Analysis (EDA)

# Count customers who churned and who stayed

df_clean['Churn'].value_counts()

# Calculate churn percentage

churn_percentage = (
    df_clean['Churn'].value_counts(normalize=True) * 100
)

print(churn_percentage)


# Calculate overall churn rate

churn_rate = (
    (df_clean['Churn'] == 'Yes').sum()
    / len(df_clean)
) * 100

print("Churn Rate:", round(churn_rate, 2), "%")



# Churn का visualization
plt.figure(figsize=(7, 5))

sns.countplot(data=df_clean, x='Churn')

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.show()




#Male vs female churn
sns.countplot(
    data=df,
    x="gender",
    hue="Churn"
)

plt.title("Customer Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Customer Count")

plt.show()




#churn rate each gender
gender_churn_rate = pd.crosstab(
    df["gender"],
    df["Churn"],
    normalize="index"
) * 100

print(gender_churn_rate)




# Contract Type vs Churn
sns.countplot(
    data=df,
    x="Contract",
    hue="Churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Customer Count")

plt.xticks(rotation=20)
plt.show()





#contract_churn_rate 
contract_churn_rate = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print(contract_churn_rate)




#Tenure analysis
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=20,
    kde=True
)

plt.title("Customer Churn by Tenure")
plt.xlabel("Tenure (Months)")
plt.ylabel("Customer Count")

plt.show()




# Tenure Groups

df["Tenure_Group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=[
        "0-12 Months",
        "13-24 Months",
        "25-48 Months",
        "49-72 Months"
    ]
)



tenure_churn_rate = pd.crosstab(
    df["Tenure_Group"],
    df["Churn"],
    normalize="index"
) * 100

print(tenure_churn_rate)