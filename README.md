# Stock Trading Strategy

## I. Background

From early 2020, the COVID-19 outbreak has disrupted the entire retail brokerage and investment industry and ways of working, in a manner we have not expected previously. Amid the stay-at-home order, stimulus packages issued by the government, and zero commission fees across most of the trading platforms, multitudes of younger investors jumped into the stock market. The impact of the new generation of retail traders has started to play a role in the directional flow of the stock market in the United States. Major stock trading platforms have reported record growth of both new funded accounts and daily average revenue trades (DARTs) in 2020. Robinhood, one of the leading investing apps announced that three million new customer accounts were added in the first quarter and there were 4.3 million DARTs in June. Yet the younger generation, many of them first-time investors, are not well-educated in investment, and sometimes arbitrary. Reading financial indicators, tracking events, and evaluating a company’s performance can be overwhelming. Under this situation, the team intended to develop a framework that takes advantage of quantamental analysis to help first-time investors make wise investment decisions.

## II. Methodology

### A. Quantamental Analysis

Quantamental combines two types of investment strategies- quantitative and fundamental, where the two are both mainstream approaches that investment firms rely on. The former focuses on statistical methods and data analytics to track the price movements based on current and historical trading data, while the latter involves studying financial indicators to evaluate the growth of a stock. However, both types of analysis have limitations and weaknesses when performed in isolation. According to a study of 4000 hedge funds by Aurum, quantamental strategies achieved an average rate of return of 14.06% in 2019, compared to 4.46% for quantitative strategies and 9.61% for fundamental-only strategies. Therefore, given the strong potential, quantamental analysis serves as the backbone of the project.

### B. Machine Learning

Machine learning is one of the favorable analytical tools in the fintech industry with its predictive power and ability to decision making. In this project, the team has implemented supervised learning methods to predict future stock returns and then to construct a portfolio based on stocks’ fundamentals.

### C. Optimization

The team applies two modern optimization approaches to assign weights to the portfolio in order to maximize the expected return in a trading period- Monte Carlo Simulation and Nelder-Mead Simplex Algorithm.

## III. Stage One - Portfolio Construction

### A. Data Cleaning

The raw data is extracted from Quandl and Yahoo Finance. The data from Quandl stores over 5,000 companies’ 109 quarterly financial indicators in ten years, between 3/31/2010 and 3/31/2020. The data from Yahoo Finance records the prices of 2,734 stocks of each trading day between 3/31/2010 and 11/11/2020. The sample data from both sources is displayed below (Fig.1 and Fig.2).

![img](https://lh6.googleusercontent.com/4oY4K24tCEZKx0bF_CH0Qynm4QWvvQDj_0nrN6r2zSQPLgu2qW4q0uj8duPwNRWCmIP6eAUJBFg2xdkjY5Jjl9m_j1xWlqbPNSP-OlZ9Fxue08Xi0eHuE3myMO5FKshQb6aUTJ90)

Fig.1 Sample Raw Data of Fundamentals from Quandl

![img](https://lh6.googleusercontent.com/5FDiyvLrDC2A8QMlwLcCBxdYubPwzQXsy_dpUs3JzNudXJpca-uML7SOMWvJPBBI7RBtpWN_a9seblLTJuRubdD95FAlts4NcbWUy4XOywMc-2kgEBHCmGO2hL-vxKbLFsz3McIX)

Fig.2 Sample Raw Data of Stock Price from Yahoo Finance

The fundamental data from Quandl contains some missing values. Since the size of the data is fairly large, and there are not too many missing values, we simply delete the rows with NAs instead of replacing them with 0 or the mean value to avoid potential bias. Also, we only keep the stocks with a market capitalization of over $10 million.

The independent variables are selected from the companies’ fundamentals, which are 25 quarterly financial indicators out of 109 originally in the raw data. These indicators are classified under four categories, which are market value, liquidity and solvency, profitability, and financial leverage. The details of 25 independent variables are listed below (Table 1).

| **Market Value (7)**           | Market Capital                                               |
| ------------------------------ | ------------------------------------------------------------ |
|                                | Total Assets                                                 |
|                                | Shareholders Equity                                          |
|                                | Enterprise Value                                             |
|                                | Price-to-Book (PB)                                           |
|                                | Price-to-Earnings (PE)                                       |
|                                | Price-to-Sales (PS)                                          |
| **Liquidity and Solvency (6)** | Free Cash Flow (FCF)                                         |
|                                | Free Cash Flow per Share                                     |
|                                | Net Cash Flow from Operations                                |
|                                | Cash and Equivalents                                         |
|                                | Working Capital                                              |
|                                | Invested Capital                                             |
| **Profitability (11)**         | Net Income                                                   |
|                                | Net Operating Income                                         |
|                                | Net Margin                                                   |
|                                | Revenue                                                      |
|                                | Earnings per Share (EPS)                                     |
|                                | Earnings before Tax (EBIT)                                   |
|                                | Earnings Before Interest Taxes & Depreciation Amortization (EBITDA) |
|                                | Return on Average Equity (ROE)                               |
|                                | Return on Average Asset (ROA)                                |
|                                | Return on Invested Capital (ROIC)                            |
|                                | Dividend Yield                                               |
| **Financial Leverage (1)**     | Debt-to-Equity                                               |

 Table.1 25 Fundamental Factors as Independent Variables

Based on the stock price data, the team first calculates the mean return of each quarter in ten years as the goal of the analysis is to see for any individual stock if the fundamentals of the last quarter can accurately predict the mean return of the next quarter. After that, we introduce a benchmark, S&P 500 Index, and then similarly, calculate the mean return of each quarter. If a stock’s quarterly mean return beats S&P 500 by 10%, the stock is classified as 1, otherwise, 0.After the preprocessing step, the cleaned dataset now contains 48,370 rows and 31 columns, as displayed in Fig.3. To perform further analysis, we randomly selected 70% of the dataset, which is 33,859 rows as the training data for model building, and used the rest of the 14,511 rows as test data for model validation.

![img](https://lh5.googleusercontent.com/8jkebhtQqT-V-esx2ntpd5u9O2YByDgF43B0SmNUqciC7AaoQ4v0MSJPHbG15KyRajYu2hbnNIMF3-KLvxPtVxwDxCB_BHZtnr9D8YKH_TnVE8RbG13LiS4SpR-5KafB-kp5JZH-)

Fig.3 Summary of the Cleaned Dataset

### B. Random Forest & Portfolio Creation

The team has performed four classification algorithms including Random Forest, Logistic Regression, Ridge Regression, Support Vector Machine (SVM), and Random Forest provides the best performance among the four classifiers. Random Forest can be considered as an improved version of single classification trees by growing a number of trees on bootstrapped training samples. The approach has three main advantages: First, it provides higher accuracy than single decision trees with low bias and moderate variance. Second, it has the power to analyze a large dataset with higher dimensionality. Third, it is easy to view the relative importance it assigns to the input features. In terms of our Random Forest results, the accuracy score is as high as 0.98, the precision score is 0.83, and the RMSE is only 0.13. As seen in Fig.4, almost every fundamental factor we select is important, especially the Price-to-Sale ratio and Price-to-Book ratio. It seems that the Dividend Yield contributes not too much to the prediction and this may result from a large majority of the public traded companies not offering a dividend to the investors. 

![img](https://lh6.googleusercontent.com/pCmT8wVqS3eawQ4ucyW9y5bVbg_uZ8t_rkhbOBbr0Z8wFQe4UD3uf42OkqvB2NupSfZybI6sLCdxuLuJAA_A-pepkiaeVzZ5iUY7thyovD546L5Jv0W3mRxkLinec5ChERw7QaN4)

Fig.4 Random Forest Output - Feature ImportanceIn addition to providing us with the feature importance, the algorithm can also help create a portfolio. Out of the stocks that outperform S&P 500 by 10% from the prediction, we reserve the top 5 stocks to construct the portfolio. The stock tickers are TWLO, BE, DKNG, SLP, and ZM, and the details of the companies are shown in Table.2.

| Stock Ticker | Company                   | Description                                                  |
| ------------ | ------------------------- | ------------------------------------------------------------ |
| NYSE: TWLO   | Twilio                    | Twilio is an American cloud communications platform as a service company that allows software developers to programmatically perform communication functions using its web service APIs. |
| NYSE: BE     | Bloom Energy              | Bloom Energy is a California-based company that manufactures and markets solid oxide fuel cells that produce electricity on-site. |
| NASDAQ: DKNG | DraftKings                | DraftKings is an American daily fantasy sports contest and sports betting operator. |
| NASDAQ: SLP  | Simulations Plus          | Simulations Plus develops simulation software for pharmaceutical and biotechnology and industrial chemicals. |
| NASDAQ: ZM   | Zoom Video Communications | Zoom Video Communications is an American communication company that provides online chat services through a cloud-based peer-to-peer software platform. |

Table.2 Portfolio of the Five Companies

In the 10-year period, our portfolio’s mean return is 19.5%. Compared to the mean return of 1.4% for the S&P 500 index, our strategy earns 18.1% more.

## IV. Stage Two - Portfolio Optimization

The team now has five stocks selected from the stage one to build up the portfolio. First, we calculate the mean weekly return of each stock based on the historical data. Then, use log return to calculate sharpe ratio. Finally, maximize the sharpe ratio to achieve portfolio optimization.

Developed by Willian Sharpe, sharpe ratio is one of the most universal risk/return measures used in finance. It allows us to quantify the relationships between return of an investment compared to its risk. A higher sharpe ratio is preferred, because it will give us a better investment performance with a comparatively smaller risk. If use a sharpe ratio to maximize portfolio return, there is one assumption that should be satisfied: the returns of the investment is normally distributed. Therefore, we checked the normality of investment return before applying a sharpe ratio. In our project, we set risk-free rate as zero. 

![img](https://lh3.googleusercontent.com/ZNYHBHOnFBj8bmJCBQ7DOc2VqdDItVM23J7KIkFwEPdWkEGdwUaI2BnqJZlyPbcwWd1RmYmKv9bwTkY4fcgsNBXj0Dg7NYMp4ObWt9GpDsUZRubXx2imMPwr99hnFeg7lDD6r-y3)

Fig.5 Sharpe Ratio Formula

### A. Method 1: Monte Carlo Simulation

In this method, we randomly assigned weights to the five stocks, and found one that gives us the best Sharpe Ratio. By doing so, we randomly assign weights to stocks in our portfolio, and then calculate the weekly return. This allows us to calculate the Sharpe Ratio for many randomly selected allocations. Then plot the allocation on a chart (Fig.6) that displays the expected return vs. the volatility, colored by Sharpe Ratio. All the points are sharpe ratio. The optimal sharpe ratio is the one we are using to achieve portfolio optimization.

Random trying gives us a clear view and better understanding of the relationship between expected return and expected volatility. The maximum return does not give us an optimal sharpe ratio, which means it usually comes along with a comparatively higher risk, while the optimal sharpe ratio can lower the risk and while getting a desirable return. Although we can try as many times as we want, randomly tring is time consuming, inefficient and also not scientifically accurate.

![img](https://lh3.googleusercontent.com/wKMmOhxmGxDgGxOJSUImA-lvMCEJT-gTMe0Ua5YjVWSQixlir98uNYTA5FlwRxu7Y6SETnWY0IGdUjD25E1N7rK4PnH61AL568q1IoEIpXZ9ie1fHWLDa83_FN91sDbg_biPAOGn)

Fig.6 Monte Carlo Simulation Output

### B. Method 2:Nelder-Mead Simplex algorithm

The idea of Nelder-Mead Simplex algorithm in portfolio optimization is to minimize the negative Sharpe Ratio, in another way, to maximize the sharpe ratio. In particular, we are using SciPy’s built-in optimization algorithm to calculate the optimal weight for portfolio allocation, optimized for sharpe ratio. The scipy.optimize package provides several commonly used optimization algorithms. The one we are applying is Nelder-Mead Simplex algorithm, which is a commonly applied numerical method used to find the minimum or maximum of an objective function in a multidimensional space.

We first define a function to calculate return, volatility, and the sharpe ratio. Before using SciPy’s algorithm, we need to define a function to get a negative sharpe ratio, adding constraint to weights which range from 0 to 1.Finally, just call the method “ minimize” and get the optimal results. 

## V. Stage Three - Backtesting

Backtesting is an excellent way to check the validity of a trading model. By bringing in historical data into the model, we can test the model and understand the model's performance in the given historical period. The backtesting steps are as follows: set backtesting period, specify backtesting strategy, set initial fund and trading day, set a benchmark and run backtesting, display and visualize backtesting performance.

### A. Set Backtesting Period

The team set the starting date of the backtesting as July 25, 2019, and the ending date as November 10, 2020.

The initial date of TWLO's listing is July 25, 2019. Therefore, the daily adjusted prices of the selected five stocks can be fully obtained after this date. November 10, 2020 is the date for our team to write the backtesting code. The selected five stocks' daily adjusted prices in the testing time range were imported from Yahoo Finance into Python as the testing dataset. There are 68 trading weeks in the testing period in total. 329 daily prices were obtained for each stock.

### B. **Backtesting Strategy**

Strategy plays a decisive role in the investment. The strategy of this model was generated as follows:

In the first week, the team manually assigned the average weight to each stock. In other words, we invested in each stock by 20% of initial cash. Starting from the second week, the program calculated the quota weights on the trading day every week based on each stock's price movements over the last 20 trading days for achieving an optimized portfolio sharpe ratio. On every trading day, the team cleared out all stocks then re-purchased them according to the new assigned weight. Therefore, the quota weights were renewed weekly, and stock clearance and re-buying trades were run weekly as well.

This strategy aims to gain profit by investing a large proportion of capital in the bullish stocks and a small proportion or no proportion in the bearish stocks to buy the rising, avoid the falling, and reduce the risk.

### C. **Set Initial Fund and Trading Day**

The initial capital for the backtesting is $10,000. Trading day was set as every Tuesday.

### D. **Set Benchmark and Run Backtesting**

In addition to trading the portfolio based on the strategy above, the model incorporated the stock SPY as a benchmark. 

Stock SPY is designed to track the S&P 500 stock market index. It is the largest ETF (Exchange-traded fund) in the world. We invested $10,000 in the SPY at the beginning of the backtesting and observed the value change of the benchmark at the end of the backtesting. In the final, the team compared increasing ratios between the portfolio and the benchmark. If the portfolio's growth is higher than that of the benchmark, the model is effective in the backtesting period.

### E. **Backtesting Performance and Visualize**

The table below (Table.3) shows the key index of the portfolio and that of the benchmark performed in the backtesting period.

**Increase Ratio:** Backtesting result shows that the value of the portfolio has increased 313.6% in the testing period, while the benchmark value increased by 20.4%. The portfolio far outpaced the benchmark

**Annualized Rate of Return:** The index represents the expected rate of return for an investment term of one year. The portfolio has an annualized rate of return of 130.15% , which is significantly higher than that of the benchmark,19.10%.

**Sharpe Ratio:** Sharpe Ratio represents that, for each unit of total risk, how much excess return is generated. The portfolio’s sharpe ratio is 12.33% , which is higher than the benchmark of 4.01%. It means the portfolio generated more return by taking each unit of total risk.

**Volatility:** Volatility measures the riskiness of an asset. It is calculated by taking the annualized standard deviation of the daily return. The portfolio has higher volatility than the benchmark.

**Max Drawdown:** Max drawdown describes the worst scenario for a strategy, measuring the largest single drop from a peak to a trough in the value of a portfolio (before a new peak is attained).Our portfolio has a max drawdown of 34.64%, which is very close to that of the benchmark , 33.72%,indicating the portfolio takes a similar amount of risk as the benchmark does.

| **Backtesting Period**        | **2019/7/25 – 2020/11/10** |                     |
| ----------------------------- | -------------------------- | ------------------- |
| **Investment**                | **Portfolio**              | **Benchmark (SPY)** |
| **Initial Investment Amount** | $10,000                    | $10,000             |
| **Ending Value**              | $41,361                    | $12,037             |
| **Minimum Value**             | $7,000                     | $7517               |
| **Maximum Value**             | $63,286                    | $12,113             |
| **Increase Ratio**            | 313.6%                     | 20.4%               |
| **Annualized Rate of Return** | 130.15%                    | 19.10%              |
| **Annualized Volatility**     | 10.55                      | 4.76                |
| **Sharpe Ratio**              | 12.33%                     | 4.01%               |
| **Max Drawdown**              | 34.64%                     | 33.72%              |

Table.3 Backtesting Performance of the Portfolio and the Benchmark (SPY)

To further exhibit how the portfolio smartly bought the rising and avoided the falling stocks, we used Tableau to visualize the value trends of the portfolio, the benchmark, and the five individual stocks, along with the weight changes calculated by the model throughout the entire backtesting period. The charts below clearly show that as some stock prices went up significantly during the time, the model smartly assigned heavier weights to them in the same period.

![img](https://lh3.googleusercontent.com/sTmsAgNSB5YvYIVKjoczLgZrbjUfyDRv8BWtMiAPoRqmAubVsdF8OdCaPZNUMBketeHMxAG0XrLx3yXUSCIoMSakLCh4e1QSO389qxuVrJfpYG0YYphsCHa9GFgvMT2SbZC1gWlq)

Fig.7 Trends of Portfolio Value and Benchmark SPY Value

![img](https://lh3.googleusercontent.com/vgmUu9P8thMh_G0KzKZb50D7o0dLNepfL0RBippzK1lrejKLtLTspNvsqHVRQlawHyD3cWYCC1aPENBP1jaUWIMtIVRyo2VuuLUjnZZYtZuA4aOIJsNlS7GB27LoKYvBQhhitBlL)

Fig.8 Selected Five Stocks Value

![img](https://lh6.googleusercontent.com/a_KvSj7X7ZAXwnNmsYpkQov5vbXqreJJXJCjC04qNRHSbJDLHYXR4bJ6LKt-EG7FNrW1YtQWx1ynJxr7MauSC5czIYLEokpvhQ-NbPp9z32EmY8FeAtOtIHReJB0_zBR4aazeIZy)

Fig.9 Changes of Assigned Weights in Backtesting Period

## VI. **Stage Four - Live Trading** 

The live trading on Robinhood started from November 20 to December 10, with an initial amount of \$10,000. The stocks and its holding amounts follow with the results of backtests. The trading frequency is weekly on every Tuesday, one hour before market close, the team calculates a new weight of each stock, then buys and sells the related stocks on Robinhood. At the noon of December 10, the total value of the portfolio is \$14,206.29, bringing a return of ​\$4206.29, the annualized return rate is 452%. The max drawdown is 10%. The strategy missed some stock rising period due to the weekly trading frequency, which exposes one drawback of this strategy that can not catch some rapid stock price change. The screenshot of Robinhood is shown in Fig.10

![img](https://lh6.googleusercontent.com/6O_Y0lObylR7SBLyHKx_F-w6tn1yEeTLTGCxfwcg5sooJLqtkczadlNx2Yc7SiTseW5aJ5vxb6jftiqK994zJKJ1O23-B6PGpUGsb9IhjDTt77RzkmXroDydiV7py6EqUvry_T1R)

Fig.10 Live Trading Screenshot from Robinhood

## VII. **Conclusion**

### A. **Decision**

The team developed a stock trading strategy framework that provides quarterly stock selection based on stock fundamental data, portfolio stock weight optimization, portfolio backtesting and return analysis. The framework is flexible to apply at any stock trading platform and proved profitable at a 20-day live trading test.

### B. **Limitation**

The stock selection model using the fundamental data from the stock quarterly report. Compared with the stock price fluctuations, the fundamental data can better reflect the company’s true performance of profitability. But the stock selection has two major limitations, one is the portfolio return is tied to the performance of selected stock in a whole quarter. The other one is the stock selection can not track the newly IPO stock, since those stocks do not release their quarterly report. The backtest framework does not take dividends and possible stock splits into account.

### C. **Takeaway**

One alternative approach for stock selection is to switch the data from fundamental data to price and volume-based information to build the model. One considerable limitation would be low data variables, there is a method to ease the concern is to create reasonable indexes such as 5-day average price data. By using the daily-update, price-based stock data, the stock selection model can involve more powerful models like Long Short-Term Memory models in the deep learning field for better prediction results. Furthermore, more assets like Bitcoin or ETF become trackable and add into the portfolio. Finally, the trading frequency is mandatory once a week. By adjusting the trading frequency based on backtesting results could lead to a different return.



## **References**

1. Zhong, X., & Enke, D. (2017). A comprehensive cluster and classification mining procedure for daily stock market return forecasting. Neurocomputing, 267, 152-168. doi:10.1016/j.neucom.2017.06.010

2. Vijh, M., Chandola, D., Tikkiwal, V. A., & Kumar, A. (2020). Stock Closing Price Prediction using Machine Learning Techniques. Procedia Computer Science, 167, 599-606. doi:10.1016/j.procs.2020.03.326

3. Lee, T. K., Cho, J. H., Kwon, D. S., & Sohn, S. Y. (2019). Global stock market investment strategies based on financial network indicators using machine learning techniques. Expert Systems with Applications, 117, 228-242. doi:10.1016/j.eswa.2018.09.005

4. Klebnikov, S. (2020, August 17). Robinhood Valuation Soars To $11.2 Billion With New Funding And Record Growth. Retrieved December 15, 2020, from https://www.forbes.com/sites/sergeiklebnikov/2020/08/17/robinhood-valuation-soars-to-112-billion-with-new-funding-and-record-growth/?sh=6f9f19dc663d
5. Rega, S. (2020, October 07). How Robinhood and Covid opened the floodgates for 13 million amateur stock traders. Retrieved December 15, 2020, from https://www.cnbc.com/2020/10/07/how-robinhood-and-covid-introduced-millions-to-the-stock-market.html
