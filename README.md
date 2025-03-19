# Sentiment-Based-Stock-Market-Prediction---XGBoost-LSTM-NLP-Data-Analytics

Successful investment strategies need to be ahead of stock market movements. Machine learning paves the way for the development of financial theories that can forecast those movements. In this work an application of the Triple-Barrier Method and Meta-Labeling techniques is explored with XGBoost for the creation of a sentiment-based trading signal on the S&P 500 stock market index. The results confirm that sentiment data have predictive power, but a lot of work is to be carried out prior to implementing a strategy.


Stock market predictions are complex due to market volatility and the influence of multiple variables. This project addresses these challenges by:

Developing models to predict stock price movements.
Classifying market trends using technical indicators.
Providing an interactive platform for analyzing and visualizing predictions.
Features
Stock Price Prediction: Predicts price movements with high accuracy using LSTM and regression models.
Market Trend Classification: Classifies trends with models like RandomForest and XGBoost.
Interactive Demo: Streamlit-based interface for real-time model predictions.
Financial Indicators: Uses EMA, MACD, RSI, and Bollinger Bands for feature engineering.
Tools and Technologies
Programming Language: Python
Libraries:
Pandas, NumPy: Data manipulation and computation.
Matplotlib, Seaborn: Data visualization.
Scikit-learn: Machine learning and preprocessing.
XGBoost: Gradient boosting models.
Imbalanced-learn (SMOTE): Addressing class imbalances.
TA (Technical Analysis): Financial indicators like EMA, MACD, and Bollinger Bands.
Modeling Techniques:
LSTM
RandomForestClassifier
XGBoostClassifier
GradientBoostingClassifier
Logistic Regression
SVM
Visualization Tools: Matplotlib, Seaborn, Streamlit.
Data Sources: Yahoo Finance (yfinance).
Workflow
Data Collection and Preprocessing:
Fetch data using yfinance.
Clean data and handle missing values.
Feature Engineering:
Create technical indicators (EMA, MACD, RSI).
Add custom features like Bollinger Bands and Stochastic Oscillator.
Exploratory Data Analysis (EDA):
Visualize relationships and trends using correlation matrices, line charts, and scatterplots.
Model Development:
Implement classifiers for trend prediction.
Optimize models using TimeSeriesSplit and SMOTE for resampling.
Interactive Dashboard:
Build a user-friendly interface using Streamlit for real-time model predictions and visualizations.
