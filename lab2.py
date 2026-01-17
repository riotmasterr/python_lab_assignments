###A1
##1
import pandas as pa
import numpy as nu

def load_data():
    
    df = pa.read_excel("Lab Session Data.xlsx", sheet_name=0)

   
    x = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values

    return x,y 


def find_cost(x, y):
    
    x_inv = nu.linalg.pinv(x)

  
    cost = x_inv @ y
    return cost

def main():
    x, y = load_data()

    
    rank = nu.linalg.matrix_rank(x)

   
    cost = find_cost(x, y)

    print("feature matrix x:\n", x)
    print("\noutput vector y:\n", y)
    print("\nrank of the feature matrix:", rank)

    print("\nestimate cost of the products:")
    print("Candies (Rs per unit):", cost[0])
    print("Mangoes (Rs per kg):", cost[1])
    print("Milk Packets (Rs per packet):", cost[2])

if __name__ == "__main__":

    main()


###A2
import pandas as pd
import numpy as np

def load_data():
    
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name=0)
    df["Class"] = df["Payment (Rs)"].apply(
        lambda x: "RICH" if x > 200 else "POOR"
    )
   
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values

    return X, y

def find_product_cost(X, y):
    
    X_pinv = np.linalg.pinv(X)
  
  
    cost = X_pinv @ y
    return cost
def main():
    X, y = load_data()
    rank = np.linalg.matrix_rank(X)
    cost = find_product_cost(X, y)
    print("Feature Matrix X:\n", X)
    print("\nOutput Vector y:\n", y)
    print("\nRank of Feature Matrix:", rank)

    print("\nEstimated Cost of Each Product:")
    print("Candies (Rs per unit):", cost[0])
    print("Mangoes (Rs per kg):", cost[1])
    print("Milk Packets (Rs per packet):", cost[2])




if __name__ == "__main__":
    main()

###A3
import pandas as pa
import numpy as nu
import time
def load_data():
 df = pa.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")
 price = df.iloc[:, 3].values
 return df, price
def manual_mean(data):
 total = 0
 for val in data:
  total += val
 return total / len(data)
def manual_variance(data):
 mean_val = manual_mean(data)
 total = 0
 for val in data:
  total += (val - mean_val) ** 2
 return total / len(data)
def time_function(func, data):
 start = time.time()
 func(data)
 end = time.time()
 return end - start
def main():
 df, price = load_data()
 print("Manual Mean:", manual_mean(price))
 print("Manual Variance:", manual_variance(price))
 print("NumPy Mean:", nu.mean(price))
 print("NumPy Variance:", nu.var(price))
 manual_time = sum(time_function(manual_mean, price) for
_
 in range(10)) / 10
 numpy_time = sum(time_function(nu.mean, price) for
_
 in range(10)) / 10
 print("Average Manual Mean Time:", manual_time)
 print("Average NumPy Mean Time:", numpy_time)
if __name__  == "_main_":
 main()
#A3.3.a
import pandas as pd
file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="IRCTC Stock Price")
wednesday_prices = df[df["Day"] == "Wed"]["Price"]
print("Wednesday Sample Mean:", wednesday_prices.mean())
#A3.3.b
import pandas as pd
file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="IRCTC Stock Price")
april_prices = df[df["Month"] == "Apr"].iloc[:, 3]
print("April's Sample Mean is", april_prices.mean())
#A.3.5
import pandas as pd
file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="IRCTC Stock Price")
losses = list(filter(lambda x: x < 0, df["Chg%"]))
prob_loss = len(losses) / len(df)
print("probability of loss is", prob_loss)
#A.3.6
import pandas as pd
file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="IRCTC Stock Price")
profit_wed = df[(df["Day"] == "Wed") & (df["Chg%"] > 0)]
print("Probability of Profit on Wednesday:",
len(profit_wed) / len(df))
#A.3.7
import pandas as pd
import matplotlib.pyplot as plt
file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="IRCTC Stock Price")
plt.scatter(df["Day"], df["Chg%"])
plt.xlabel("Day of Week")
plt.ylabel("Change %")
plt.title("Scatter Plot of Chg% vs Day")
plt.show()

###A4
import pandas as pd
import numpy as np


def load_data(file_path, sheet_name):
    dataframe = pd.read_excel(file_path, sheet_name=sheet_name)
    return dataframe



def get_numerical_columns(dataframe):
    return dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()



def min_max_normalization(dataframe, numerical_columns):
    normalized_df = dataframe.copy()
    for column in numerical_columns:
        min_val = dataframe[column].min()
        max_val = dataframe[column].max()
        normalized_df[column] = (dataframe[column] - min_val) / (max_val - min_val)
    return normalized_df


def z_score_standardization(dataframe, numerical_columns):
    standardized_df = dataframe.copy()
    for column in numerical_columns:
        mean_val = dataframe[column].mean()
        std_val = dataframe[column].std()
        standardized_df[column] = (dataframe[column] - mean_val) / std_val
    return standardized_df



def main():
    df = load_data("Lab Session Data.xlsx", "thyroid0387_UCI")
    numerical_cols = get_numerical_columns(df)

    min_max_df = min_max_normalization(df, numerical_cols)
    z_score_df = z_score_standardization(df, numerical_cols)

    print("Min-Max Normalized Data (first 5 rows):\n")
    print(min_max_df[numerical_cols].head())

    print("\nZ-score Standardized Data (first 5 rows):\n")
    print(z_score_df[numerical_cols].head())


main()

###A5
import pandas as pd
import numpy as np
file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="thyroid0387_UCI")
numeric_df = df.select_dtypes(include=[np.number])
binary_df = numeric_df.fillna(0)
binary_df = binary_df.applymap(lambda x: 1 if x > 0 else 0)
v1 = binary_df.iloc[0].values
v2 = binary_df.iloc[1].values
f11 = f10 = f01 = f00 = 0
for i in range(len(v1)):
 if v1[i] == 1 and v2[i] == 1:
    f11 += 1
 elif v1[i] == 1 and v2[i] == 0:
  f10 += 1
 elif v1[i] == 0 and v2[i] == 1:
  f01 += 1
 elif v1[i] == 0 and v2[i] == 0:
  f00 += 1
# jaccard coefficient
jaccard = f11 / (f11 + f10 + f01)
# simple matching coefficient
smc = (f11 + f00) / (f11 + f10 + f01 + f00)
print("f11:", f11)
print("f10:", f10)
print("f01:", f01)
print("f00:", f00)
print("jaccard coefficient value is", jaccard)
print("simple matching coefficient value is", smc)

###A6
import pandas as pd
import numpy as np


# Load dataset

def load_data(file_path, sheet_name):
    dataframe = pd.read_excel(file_path, sheet_name=sheet_name)
    return dataframe



# Separate categorical and numerical attributes


def separate_columns(dataframe):
    categorical_columns = dataframe.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    return categorical_columns, numerical_columns



# Check for outliers using IQR

def has_outliers(dataframe, column_name):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = dataframe[
        (dataframe[column_name] < lower) | (dataframe[column_name] > upper)
    ]
    return not outliers.empty



# Impute missing values

def impute_missing_values(dataframe, categorical_columns, numerical_columns):
    imputation_details = []

    for column in numerical_columns:
        if dataframe[column].isnull().sum() > 0:
            if has_outliers(dataframe, column):
                replacement = dataframe[column].median()
                dataframe[column].fillna(replacement, inplace=True)
                imputation_details.append((column, "Median"))
            else:
                replacement = dataframe[column].mean()
                dataframe[column].fillna(replacement, inplace=True)
                imputation_details.append((column, "Mean"))

    for column in categorical_columns:
        if dataframe[column].isnull().sum() > 0:
            replacement = dataframe[column].mode()[0]
            dataframe[column].fillna(replacement, inplace=True)
            imputation_details.append((column, "Mode"))

    return dataframe, imputation_details



def main():
    df = load_data("Lab Session Data.xlsx", "thyroid0387_UCI")
    categorical_cols, numerical_cols = separate_columns(df)

    df, details = impute_missing_values(df, categorical_cols, numerical_cols)

    print("Imputation Summary:\n")
    for column, method in details:
        print(f"{column}: filled using {method}")

    print("\nMissing Values After Imputation:\n")
    print(df.isnull().sum())


main()
###A7
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="thyroid0387_UCI")

#  first 20 observation vectors
df_20 = df.iloc[:20]

#  numerical columns
numeric_df = df_20.select_dtypes(include=[np.number]).fillna(0)

# converting - binary for JC , SMC
binary_df = numeric_df.apply(lambda col: col.map(lambda x: 1 if x > 0 else 0))


# similarity functions

def jaccard(v1, v2):
    f11 = f10 = f01 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
    return f11 / (f11 + f10 + f01)

def smc(v1, v2):
    f11 = f10 = f01 = f00 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
        elif v1[i] == 0 and v2[i] == 0:
            f00 += 1
    return (f11 + f00) / (f11 + f10 + f01 + f00)

def cosine(v1, v2):
    dot = 0
    mag1 = 0
    mag2 = 0
    for i in range(len(v1)):
        dot += v1[i] * v2[i]
        mag1 += v1[i] ** 2
        mag2 += v2[i] ** 2
    return dot / ((mag1 * 0.5) * (mag2 * 0.5))

#  similarity matrices

jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))
cos_matrix = np.zeros((20, 20))

# calculate similarity matrix

for i in range(20):
    for j in range(20):
        jc_matrix[i][j] = jaccard(binary_df.iloc[i].values,
                                  binary_df.iloc[j].values)

        smc_matrix[i][j] = smc(binary_df.iloc[i].values,
                               binary_df.iloc[j].values)

        cos_matrix[i][j] = cosine(numeric_df.iloc[i].values,
                                  numeric_df.iloc[j].values)

# heatmap plotted

plt.imshow(jc_matrix, cmap='hot')
plt.colorbar()
plt.title("jaccard values heatmap")
plt.show()

plt.imshow(smc_matrix, cmap='hot')
plt.colorbar()
plt.title("simple Matching value heatmap")
plt.show()

plt.imshow(cos_matrix, cmap='hot')
plt.colorbar()
plt.title("cosine similarity heatmap")
plt.show()

###A8
import pandas as pa
import numpy as nu

def load_data():
    
    df = pa.read_excel("Lab Session Data.xlsx", sheet_name=0)

   
    x = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values

    return x,y 


def find_cost(x, y):
    
    x_inv = nu.linalg.pinv(x)

  
    cost = x_inv @ y
    return cost

def main():
    x, y = load_data()

    
    rank = nu.linalg.matrix_rank(x)

   
    cost = find_cost(x, y)

    print("feature matrix x:\n", x)
    print("\noutput vector y:\n", y)
    print("\nrank of the feature matrix:", rank)

    print("\nestimate cost of the products:")
    print("Candies (Rs per unit):", cost[0])
    print("Mangoes (Rs per kg):", cost[1])
    print("Milk Packets (Rs per packet):", cost[2])

if __name__ == "__main__":

    main()
import pandas as pa
import numpy as nu

def load_data():
    df = pa.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values
    return X, y

def classify_customers(y):
    labels = nu.where(y > 200, "rich", "poor")
    return labels

def main():
    X, y = load_data()
    labels = classify_customers(y)

    print("Customer classified based on  there payment:")
    print(labels)

if __name__ == "__main__":

    main()


###A5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = "Lab Session Data.xlsx"
df = pd.read_excel(file, sheet_name="thyroid0387_UCI")

#  first 20 observation vectors
df_20 = df.iloc[:20]

#  numerical columns
numeric_df = df_20.select_dtypes(include=[np.number]).fillna(0)

# converting - binary for JC , SMC
binary_df = numeric_df.apply(lambda col: col.map(lambda x: 1 if x > 0 else 0))


# similarity functions

def jaccard(v1, v2):
    f11 = f10 = f01 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
    return f11 / (f11 + f10 + f01)

def smc(v1, v2):
    f11 = f10 = f01 = f00 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
        elif v1[i] == 0 and v2[i] == 0:
            f00 += 1
    return (f11 + f00) / (f11 + f10 + f01 + f00)

def cosine(v1, v2):
    dot = 0
    mag1 = 0
    mag2 = 0
    for i in range(len(v1)):
        dot += v1[i] * v2[i]
        mag1 += v1[i] ** 2
        mag2 += v2[i] ** 2
    return dot / ((mag1 * 0.5) * (mag2 * 0.5))

#  similarity matrices

jc_matrix = np.zeros((20, 20))
smc_matrix = np.zeros((20, 20))
cos_matrix = np.zeros((20, 20))

# calculate similarity matrix

for i in range(20):
    for j in range(20):
        jc_matrix[i][j] = jaccard(binary_df.iloc[i].values,
                                  binary_df.iloc[j].values)

        smc_matrix[i][j] = smc(binary_df.iloc[i].values,
                               binary_df.iloc[j].values)

        cos_matrix[i][j] = cosine(numeric_df.iloc[i].values,
                                  numeric_df.iloc[j].values)

# heatmap plotted

plt.imshow(jc_matrix, cmap='hot')
plt.colorbar()
plt.title("jaccard values heatmap")
plt.show()

plt.imshow(smc_matrix, cmap='hot')
plt.colorbar()
plt.title("simple Matching value heatmap")
plt.show()

plt.imshow(cos_matrix, cmap='hot')
plt.colorbar()
plt.title("cosine similarity heatmap")
plt.show()

###A9
import pandas as pd
import numpy as np

FILE_NAME = "Lab_Session_Data.xlsx"
SHEET_NAME = "thyroid0387_UCI"

# ------------------ Load Data ------------------ #
df = pd.read_excel(FILE_NAME, sheet_name=SHEET_NAME)

# ------------------ Identify Attributes to Normalize ------------------ #
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print("Numerical attributes (need normalization):")
print(numerical_cols)

print("\nCategorical attributes (no normalization needed):")
print(categorical_cols)

# ------------------ Handle Missing Values (optional but recommended) ------------------ #
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# ------------------ Min-Max Normalization ------------------ #
minmax_df = df.copy()
for col in numerical_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    if max_val - min_val != 0:
        minmax_df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        minmax_df[col] = 0  # if constant column

# ------------------ Z-score Standardization ------------------ #
zscore_df = df.copy()
for col in numerical_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val != 0:
        zscore_df[col] = (df[col] - mean_val) / std_val
    else:
        zscore_df[col] = 0  # if constant column

# ------------------ Save Normalized Data ------------------ #
minmax_df.to_excel("thyroid_minmax_normalized.xlsx", index=False)
zscore_df.to_excel("thyroid_zscore_standardized.xlsx", index=False)

print("\n✅ Normalized datasets created successfully!")
print("1) thyroid_minmax_normalized.xlsx")
print("2) thyroid_zscore_standardized.xlsx")