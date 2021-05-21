# ETF Predictions

## Introduction

An exchange traded fund (ETF) is a fund that owns multiple underlying securities. A popular example is the SPY ETF which tracks the S&P 500 index which itself tracks the performance of the 500 largest companies in the United States. We consider 10 such ETF's (SPY, OIH,GLD, IWM, IYR, TLT, FXE, EEM, LQD, TIP, TLT). We will try to predict the directional change of the ETF price over a given horizon

## Objectives

We will have two objectives in this work
 - Predict the daily directional price changes of a given ETF (up or down). We will mainly try to quantigy if we do better than trying to predict the directional returns of a random walk through hypothesis testing.
 - Predict the directional change of an ETF over 1, 5 and 10 days by combining the features of all ETF's in a single data frame. See if our algorithms perform better than predicting the returns of a random walk.


## Conclusion and Future Work

### Daily Analysis

We find a signal in **SPY, OIH, IWM, IYR, LQD**. In most of the cases RF or XGB algorithm gives a positive signal.

![Alt text](/relative/path/to/img.jpg?raw=true "Optional Title")

However before making such a bold claim it's important to remember that our analysis only suggests that we might be able to get better accuracy compared to a random ETF, however our returns could be worse compared to zero. Also we only consider 5 different random ETF's. It's possible that considering more, we would get a p-value above our threshold of 0.05. A more detailed statistical analysis is needed to address this question. It is of interest why this method works on certain ETF's and not others and should be investigated further.

Finally we need to implement this strategy in live markets to see it's potential. We will implement this in the future. For future work we will consider following improvements.
Increase horizon size and combine ETF's together, keeping the features the same.
- Macroeconomic data such as employment numbers, retail sales, industrial production, consumer confidence etc.
- Use NLP to scan financial news headlines and generate features (This is harder than it sounds because each ETF represents a collection of different companies which different news sources can affect)
- Try different algorithms (Neural Nets, Stacking etc.)
- If a significant signal is found, try running the algorithm on active markets

### Combined ETF Analysis

When we combine the ETF's we see a signal even for horizon of one day for most of our ETF's.  For SPY however as we increased the horizon size for our predictions, the predictions got worse with no prediciton compared to the random ETF's for horizon size of 10. This is contrary to what's shown in Liew and Mayster. It is not clear why we reach different conclusions. A difference could be because they complain that their information gain to a randomly generated distribution from univariate variables, but we compare it to a random walk which should be a better comparison in theory. This needs to be investigated further.


Increasing horizon size could give a larger gain for most ETF's however it's important to remember once again that our analysis only suggests that we might be able to get better accuracy compared to a random ETF, however our returns could be worse compared to zero. Increasing horizon size could also mean lower overall returns, even for higher accuracy since profits compund. The ideal horizon should be investigated. We should also see if this stands the test of live markets.


## Blog

https://aghalsasi-datascience.blogspot.com/2021/02/etf-predictions.html

## Navigating the repository

The main analysis and presentation notebook is ETF_predictions. The Analysis ans Analysis_LW are auxillary notebooks to run our ML algorithms on our data and generate a signal. The generated accuracy scores are stored in the folder results. The auxillary functions are defined in Auxillary_Functions, Classification_Functions, Create_Features and Reproducing_Liew_Mayster.