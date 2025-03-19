# Sentiment-Based-Stock-Market-Prediction---XGBoost-LSTM-NLP-Data-Analytics

Successful investment strategies need to be ahead of stock market movements. Machine learning paves the way for the development of financial theories that can forecast those movements. In this work an application of the Triple-Barrier Method and Meta-Labeling techniques is explored with XGBoost for the creation of a sentiment-based trading signal on the S&P 500 stock market index. The results confirm that sentiment data have predictive power, but a lot of work is to be carried out prior to implementing a strategy.


# Stock market predictions are complex due to market volatility and the influence of multiple variables. This project addresses these challenges by:
1 Developing models to predict stock price movements.
2 Classifying market trends using technical indicators.
3 Providing an interactive platform for analyzing and visualizing predictions.

# Features
1 Stock Price Prediction: Predicts price movements with high accuracy using LSTM and regression models.
2 Market Trend Classification: Classifies trends with models like RandomForest and XGBoost.
3 Interactive Demo: Streamlit-based interface for real-time model predictions.
4 Financial Indicators: Uses EMA, MACD, RSI, and Bollinger Bands for feature engineering.

# Tools and Technologies
1 Programming Language: Python

# Libraries:
1 Pandas, NumPy: Data manipulation and computation.
2 Matplotlib, Seaborn: Data visualization.
3 Scikit-learn: Machine learning and preprocessing.
4 XGBoost: Gradient boosting models.
5 Imbalanced-learn (SMOTE): Addressing class imbalances.
6 TA (Technical Analysis): Financial indicators like EMA, MACD, and Bollinger Bands.

# Modeling Techniques:
1 LSTM
2 RandomForestClassifier
3 XGBoostClassifier
4 GradientBoostingClassifier
5 Logistic Regression
6 SVM

# Visualization Tools: Matplotlib, Seaborn, Streamlit.
1 Data Sources: Yahoo Finance (yfinance).

# Workflow
1 Data Collection and Preprocessing:
Fetch data using yfinance.
Clean data and handle missing values.
2 Feature Engineering:
Create technical indicators (EMA, MACD, RSI).
Add custom features like Bollinger Bands and Stochastic Oscillator.
3 Exploratory Data Analysis (EDA):
Visualize relationships and trends using correlation matrices, line charts, and scatterplots.
4 Model Development:
Implement classifiers for trend prediction.
Optimize models using TimeSeriesSplit and SMOTE for resampling.
5 Interactive Dashboard:
Build a user-friendly interface using Streamlit for real-time model predictions and visualizations.
