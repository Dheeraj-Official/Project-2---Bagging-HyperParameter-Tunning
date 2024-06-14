# Import necessary libraries
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Set random seed for reproducibility
random.seed(20)
np.random.seed(20)

# ------------------
# Data Generation and Plotting
# ------------------

# Generate data
x = 8 * np.random.rand(100) - 4 + np.random.randint(-1, 2, 100)
y = -0.02 * x * (x + 4) * (x - 4) * (x + 2) * (x + 3) + np.random.randn(100) * 2

# Plot data
def plot_data(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="yellow", linewidths=1.3, edgecolors="black", s=50)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Scatter plot of the generated data")
    st.pyplot(fig)

# ------------------
# Sidebar and User Input
# ------------------

# Sidebar for user input
def sidebar_input():
    st.sidebar.header("Bagging Regressor - By Dheeraj")
    estimator_choice = st.sidebar.selectbox(
        "Select base estimator",
        ['Decision Tree', 'SVM', 'Linear Regression']
    )
    n_estimators = st.sidebar.number_input(
        "Enter Number of estimators",
        min_value=1,
        value=50,
        step=1
    )
    max_samples = st.sidebar.slider(
        "Max Samples",
        min_value=1,
        max_value=150,
        value=25
    )
    bootstrap = st.sidebar.radio(
        "Bootstrap Settings",
        (True, False)
    )
    return estimator_choice, n_estimators, max_samples, bootstrap

# ------------------
# Model Training and Evaluation
# ------------------

# Train and evaluate selected estimator
def train_and_evaluate(estimator_choice, n_estimators, max_samples, bootstrap):
    if st.sidebar.button("Run Algorithm"):
        # Select the base estimator
        if estimator_choice == 'Decision Tree':
            estimator = DecisionTreeRegressor()
        elif estimator_choice == 'SVM':
            estimator = SVR()
        elif estimator_choice == 'Linear Regression':
            estimator = LinearRegression()

        # Fit base estimator
        estimator.fit(x.reshape(-1, 1), y)

        # Predict y values
        x_test = np.linspace(-5, 5, 100).reshape(-1, 1)
        y_pred = estimator.predict(x_test)

        # Calculate cross-validation score
        score1 = round(np.mean(cross_val_score(estimator, x.reshape(-1, 1), y, cv=10)), 3)

        # Plot base estimator results
        plot_results(x, y, x_test, y_pred, f"{estimator_choice} Regression", score1)

        # Bagging Regression
        per_sample = max_samples / x.shape[0]
        clf2 = BaggingRegressor(
            estimator=estimator,
            n_estimators=n_estimators,
            max_samples=per_sample,
            bootstrap=bootstrap
        )
        clf2.fit(x.reshape(-1, 1), y)
        y_bagged_pred = clf2.predict(x_test)

        # Calculate cross-validation score for bagging
        score2 = round(np.mean(cross_val_score(clf2, x.reshape(-1, 1), y, cv=10)), 3)

        # Plot Bagging results
        plot_results(x, y, x_test, y_bagged_pred, "Bagging Regression", score2)

# Plotting function for results
def plot_results(x, y, x_test, y_pred, title, score):
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="yellow", linewidths=1.3, edgecolors="black", s=50, label="Data")
    ax.plot(x_test, y_pred, color="red", linewidth=1.5, label=f"{title} - Cross Val Score: {score}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} - Cross Val Score: {score}")
    ax.legend()
    st.pyplot(fig)

# ------------------
# Main Function to Run the App
# ------------------

def main():
    st.title("Regression Analysis with Streamlit")
    plot_data(x, y)
    estimator_choice, n_estimators, max_samples, bootstrap = sidebar_input()
    train_and_evaluate(estimator_choice, n_estimators, max_samples, bootstrap)

if __name__ == "__main__":
    main()
