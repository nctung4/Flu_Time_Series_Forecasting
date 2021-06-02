# Flu Time Series Forecasting

This is a time series forecasting project. I wanted to make a model which can be applied to flu time series forecasting. I downloaded the data from WHO public database. The data is ranging from 2010.01.01 to 2020.06.30, and it is focused on the European WHO region. It is a weekly data. I created the models in Rstudio, and for deep learning models (LSTM) I used python. The R code is [here](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast.md), and the python notebook is [here](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/LSTM_model.ipynb).

### Project Overview
* Downloaded 10 years time series data from WHO database.
* Preprocessed the time series data in R and Python.
* Analysed the data with decomposition, seasonality check, peak detection etc.
* Implemented 3 models in R. (STLF, ARIMA, TBATS)
* Implemented LSTM models in Python. (Vanilla LSTM, Stacked LSTM, Bidirectional LSTM, Stacked LSTM with relu activation function)
 
**Final summary table of the model performance:**

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/plot/df_summary.png)

### Resources used
**Python Version:** 3.8.5 <br>
**R Version:** 4.0.2 <br>
**Python Packages:** pandas, numpy, tensorflow, matplotlib, sickit-learn, dataframe_image <br>
**R Packages:** forecast, ggplot2, dplyr, tidyr, zoo, lubridate<br>
**Requirements:** ```pip install -r requirements.txt```  
 
### Table of contents:
1. [Project Motivation](#project-motivation)
2. [Technical Aspects](#technical-aspects)
3. [Major Insights](#major-insights)
    * 3.1 [EDA](#eda)
    * 3.2 [Modeling](#modeling)
    * 3.3 [Evaluation](#evaluation)
5. [Credits](#credits)

### Project Motivation
The main purpose of this analysis and forecast is to get some information of the current situation of the flu viral infections. With the sufficient amount of data we would be able to perform a precise prediction of the flu cases in the future. With the results, we could potentially be prepared for every possible outcomes. Resources can be managed much more efficiently with knowing the precise estimates.

### Technical Aspects
* I visualized the data using **ggplot**, **matplotlib** and **Tableau**.
* I preprocessed the data in R and Python
* I optimized 3 traditional time series model in R, and 4 different LSTM architecture in Python, using **tensorflow**.

### Major Insights
In this section, I will show the major insights of this project.

#### EDA:
After implementing some data cleaning steps, I wanted to check the flu cases per countries. I analysed it in several dimensions. I checked that how many processed cases did the country had in the past 10 years, and within that how many positive cases were identified.

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/plot/New_Map_cutted.png)

The bigger the rectangle, the more processed cases does the country have. The greener the rectangle, the lesser the positive case ratio. 

It was expected that where the population is high (for instance Russia, France), there would be more processed cases. In case of the Scandinavian countries, the high number of suspicious cases are also not surprising due to the cold climate, but here the positivity rate was not significant. There is a bigger problem in countries with small populations and high positivity rate. These countries are for instance Netherlands, Lithuania, Croatia, Romania, and Greece, where the positivity rate ranges from 35% to 65%. In the highly populated countries only Spain has a more serious positivity rate.

**Flu cases weekly plot:**

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-2-1.png)

On average there are approximately 12K processed cases in a week in the European region, and within that roughly 2.5K are positive cases.

There are several types on flu viruses, the most common ones are type A and type B. 

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-3-1.png)

**Seasonality:**
The next plots shows that our dataset is seasonal.

Seasonality plot no. 1.:

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-4-1.png)

Seasonality plot no. 2.:

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-4-2.png)

Seasonality plot no. 3.:

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-4-3.png)

**Decomposition of the data:**

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-5-1.png)

#### Modeling:
In the traditional time series models, we can say that the model used all of the information available, if the residuals satifies the following assumptions:
* They are not correlated with each other.
* The mean of the residuals are 0.
* The residuals have a constant variance.
* The residuals follow a normal distribution.

In summary, we can say that the residuals must be a white noise process. We can check that with the **Ljung-box test**, whether it can be identified as white noise or not. If p>=0.05, then we can say that the assumptions are satisfied. In R we can perform this test with the checkresiduals() method from the forecast library.

**STLF model:**

Ljung-box test:

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-6-1.png)

p value: 0.7, assumtion is satisfied.

STLF model forecast:

![](https://github.com/nctung4/Flu_Time_Series_Forecasting/blob/main/Flu_forecast_files/figure-gfm/unnamed-chunk-7-1.png)

**ARIMA model:**

**TBATS model:**

**LSTM models:**
LSTM intuition

**LSTM model 1.:**
Milyen architektura
learning plot
forecasting plot

**LSTM model 2.:**
Milyen architektura
learning plot
forecasting plot

**LSTM model 3.:**
Milyen architektura
learning plot
forecasting plot

**LSTM model 4.:**
Milyen architektura
learning plot
forecasting plot

#### Evaluation:
Summary table

### Credits:
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/
* https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/
* https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
* https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
* https://otexts.com/fpp2/
* https://www.superdatascience.com/blogs/recurrent-neural-networks-rnn-the-vanishing-gradient-problem
