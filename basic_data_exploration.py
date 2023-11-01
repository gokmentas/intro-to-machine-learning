import pandas as pd

iowa_file_path = "C:\\Users\\Gokme\\Desktop\\Gokmen\\Machine Learning Projects\\intro-to-machine-learning\\train.csv"

home_data = pd.read_csv(iowa_file_path)
print(home_data.describe())