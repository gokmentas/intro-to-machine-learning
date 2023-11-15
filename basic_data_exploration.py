import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# The first number, the count, shows how many rows have non-missing values.
# The second value is the mean, which is the average. Under that, std is the standard deviation,
# which measures how numerically spread out the values are.
# To interpret the min, 25%, 50%, 75% and max values, imagine sorting each column from lowest to highest value.
# The first (smallest) value is the min. If you go a quarter way through the list,
# you'll find a number that is bigger than 25% of the values and smaller than 75% of the values.
# That is the 25% value. The 50th and 75th percentiles are defined analogously, and the max is the largest number.

# Path of the file to read
iowa_file_path = "C:\\Users\\Gokme\\Desktop\\Gokmen\\Machine Learning Projects\\intro-to-machine-learning\\train.csv"

# Read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Print summary statistics in next line
#print(home_data.describe())

# Define: What type of model will it be? A decision tree? Some other type of model?
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.

# Print the list of columns in the dataset to find the name of the prediction target
#print(home_data.columns)

# Select the target variable, which corresponds to the sales price. Save this to a new variable called y
y = home_data.SalePrice

# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# Specify the model.
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model
iowa_model.fit(X, y)

# Make predictions with the model's predict command using X as the data.
predictions = iowa_model.predict(X)

print(predictions)

# Use the head method to compare the top few predictions to the actual home values (in y) for those same homes.
print(y.head())