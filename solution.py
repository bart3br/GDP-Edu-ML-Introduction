import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def split_data(filename: str, test_size=0.25, random_state=33) -> tuple:
    """
    reading the dataset and extracting the columns with high correlation to GDP_per_capita
    (according to the pandas profiling report)
    splitting the dataset into train and test sets
    """
    #reading the dataset and extracting the columns with high correlation to GDP_per_capita
    df = pd.read_csv(filename)
    X = df[['Top_1000_Uni_Count', 'Tertiary_edu_%']].values
    Y = df['GDP_per_capita'].values
    
    #splitting the dataset into train and test data, with train:test ratio of 80:20
    #random_state is set for reproducibility
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return (df, X_train, X_test, Y_train, Y_test)


def train_linear_model(X_train, X_test, Y_train, Y_test):
    lin_model = LinearRegression()
    lin_model.fit(X_train, Y_train)
    print(f"Linear Regression model parameters: {np.round(lin_model.coef_,4)}, {np.round(lin_model.intercept_,4)}")
    MSE_lin = mean_squared_error(Y_test, lin_model.predict(X_test), squared=False)
    print(f'Root Mean Squared Error of Linear Regression model: {MSE_lin:0.3}\n')
    
    
def train_svr_model(X_train, X_test, Y_train, Y_test):
    svr_model = SVR(kernel='rbf', gamma='scale', C=1)
    svr_model.fit(X_train, Y_train)
    MSE_SVR = mean_squared_error(Y_test, svr_model.predict(X_test), squared=False)
    print(f'Root Mean Squared Error of Support Vector Machine (SVR) model: {MSE_SVR:0.3}\n')
    
    
def train_glm_model(X_train, X_test, Y_train, Y_test):
    glm_model = LinearRegression()
    gen_features = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)
    glm_model.fit(gen_features.fit_transform(X_train), Y_train)
    print(f'Generalized Linear Model parameters: {np.round(glm_model.coef_,4)}, {np.round(glm_model.intercept_,4)}')
    MSE_GLM = mean_squared_error(Y_test, glm_model.predict(gen_features.fit_transform(X_test)), squared=False)
    print(f'Root Mean Squared Error of GLM model: {MSE_GLM:0.3}\n')

def train_random_forest_model(X_train, X_test, Y_train, Y_test):
    forest_model = RandomForestRegressor()
    forest_model.fit(X_train, Y_train)
    MSE_RFR = mean_squared_error(Y_test, forest_model.predict(X_test), squared=False)
    print(f'Root Mean Squared Error of Random Forest model: {MSE_RFR:0.3}\n')
    
def train_decision_tree_model(X_train, X_test, Y_train, Y_test):
    decision_tree_model = DecisionTreeRegressor()
    decision_tree_model.fit(X_train, Y_train)
    MSE_DT = mean_squared_error(Y_test, decision_tree_model.predict(X_test), squared=False)
    print(f'Root Mean Squared Error of Decision Tree model: {MSE_DT:0.3}\n')

def make_correlation_chart(degree, data, x_label, y_label):
    poly_features = PolynomialFeatures(degree=degree, include_bias=True, interaction_only=False)
    x_poly = poly_features.fit_transform(data[x_label].values.reshape(-1,1))
    
    lin_model = LinearRegression()
    lin_model.fit(x_poly, data[y_label])
    
    x_range = np.linspace(data[x_label].min(), data[x_label].max(), 100).reshape(-1,1)
    x_range_poly = poly_features.fit_transform(x_range)
    y_pred = lin_model.predict(x_range_poly)
    
    plt.figure(figsize=(10,7))
    plt.scatter(data[x_label], data[y_label], label='data', alpha=0.7)
    plt.xlabel(x_label, fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(f'Correlation between {x_label} and {y_label}', fontsize=16)
    plt.plot(x_range, y_pred, label=f'regression line, degree: {degree}', color='tab:red')
    plt.legend()
    plt.savefig(f"plots/correlations/{x_label}_correlation_{y_label}.png")
    plt.close()

def run_solution():
    """
    splitting the dataset into train and test sets
    making correlation charts for each pair of variables chosen earlier in EDA
    creating a linear regression model, SVR model and GLM model
    training models on the train set
    predicting the values of the test set using the models
    calculating the mean squared error of the models
    """
    print("\nRunning solution...\n")
    df, X_train, X_test, Y_train, Y_test = split_data("dataset.csv", test_size=0.25, random_state=33)
    degree=2
    make_correlation_chart(degree, df, 'Top_1000_Uni_Count', 'GDP_per_capita')
    make_correlation_chart(degree, df, 'Tertiary_edu_%', 'GDP_per_capita')
    
    train_linear_model(X_train, X_test, Y_train, Y_test)
    train_svr_model(X_train, X_test, Y_train, Y_test)
    train_glm_model(X_train, X_test, Y_train, Y_test)
    train_random_forest_model(X_train, X_test, Y_train, Y_test)
    train_decision_tree_model(X_train, X_test, Y_train, Y_test)
    
    
if __name__ == "__main__":
    pass