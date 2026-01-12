
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
