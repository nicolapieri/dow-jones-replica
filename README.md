# **Dow Jones index replication with portfolio optimization methods**
The **Dow Jones Industrial Average** ("^DJI", hereinafter) is a popular stock market index that gathers 30 prominent companies listed in the U.S. stock exchanges. [Check here for general knowledge](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average) and [here for the technical details](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-dj-averages.pdf).

# The problem
Just like single stocks, stock market indexes also yield a return (positive or negative) over time. Whenever an investor wants to match exactly an index return, he/she needs to buy all the underlying stocks. This can be **considerably costly due to transaction commissions**, especially for indexes that contain hundreds of different stocks.

To overcome this problem, the investor needs to create a portfolio containing a smaller number of stocks whose aggregate return is as similar as possible to the index he/she wants to replicate. In this way, the investor can ensure a **higher overall performance simply by reducing the commission costs**.

Of course, there are limits to this kind of simplification. **The higher the number of stocks excluded from the portfolio, the higher the deviation of the portfolio return** compared to the original index (unless some of those stocks are negatively correlated with the index). This trade-off is at the core of every portfolio optimization project.

# Premise considerations
Due to some peculiarities of this project, here I list the simplifications I am taking for granted in the current version. Some of them may be considered in future versions.

*   A pool of experts changes the ^DJI stock composition periodically. For simplicity, **I will consider only the time period from the last update (August 31st 2020) to May 31st 2023**. The briefness of the time period is evident also by looking at the portfolios Betas, which are tiny. A much more valuable approach would take a wider time range (20-30 years) and would adjust the portfolio composition dynamically along with the index updates. [For the historical components of the ^DJI check here](https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average).

*   The official ^DJI close is calculated by summing all the close prices of the stocks comprised in the index and dividing by a constant, the Dow Divisor. The purpose of the Dow Divisor is to ensure that extraordinary events like stock splits, spinoffs or other structural changes, do not in themselves alter the numerical value of the ^DJI. Also, the Dow Divisor is periodically updated, **I will take for granted the latest Dow Divisor (issued on November 4th 2021) for the entirety of the project period**. As for the previous assumption, a more solid approach would dynamically change the Dow Divisor when every update was published. [For the historical Dow Divisors check here](https://www.barrons.com/market-data/market-lab?mod=md_subnav).

*   Every proper portfolio optimization strategy has periodical (monthly or trimestral) checkpoint buckets to evaluate the goodness of the replica and make some rebalancing. At every checkpoint, stock weights in the portfolio can change, so that some stocks can be excluded from the next bucket and replaced by others. **I will not make periodical buckets rebalancing, and I am assuming that all the stocks of the ^DJI should get a spot in the portfolio**. In future versions, I may prevent including a stock in the portfolio if its weight is lower than a predefined threshold, which is coherent to reduce the unnecessary commissions costs.

Lastly, I would make you notice that I am using adjusted closing prices and not just regular closing prices to try to account also splits and dividend distributions. For details see [Yahoo Finance Adj Close](https://help.yahoo.com/kb/SLN28256.html#/)

# Evaluation KPIs
I will base the replica goodness evaluation on the following KPIs, but not all of them do have the same relevance. Here I explain their interpretation.

*   **Standard Tracking Error**. It indicates the absolute difference in actual performance between the portfolio replica and the ^DJI index. It is given by the standard deviation of the differences between the returns of the portfolio and the ^DJI. Since our goal is to replicate an index and not beat the market (god forbid), I am just interested in having low Standard Tracking Errors in absolute terms.

*   **Information Ratio**. It identifies how much the portfolio replica has exceeded the ^DJI index in terms of returns. It is given by standardizing (thanks to the tracking error) the difference in performances between the portfolio and the ^DJI. Higher information ratios indicate a desired level of portfolio strategy consistency, but without a related measure of risk, it is a really naive indicator.

*   **Portfolio Active Return**. It indicates how much the portfolio replica has performed with respect to the ^DJI index, in percentage points, at the end of the holding period. It is given by computing the difference between the total portfolio holding period return and the total ^DJI holding period return. Sometimes it can be interesting to see that, in addition to the increase in performance thanks to the reduction of commission costs, there is also a fortunate (unfortunate) factor that affects positively (negatively) the overall performance just because, in the optimization procedure, good (bad) performing stocks have been included in the portfolio, rather than others.

![img.png](img.png)

# Results & learning a model
Quick premise to the results: even if some optimization methods had better tracking errors than others, **I am assuming that it is desirable to combine them all in order to have a robust strategy**. Apart from the picture, I chose to represent the results with three different tables.

* The first is also the simplest; it is just a list of the optimization methods, sorted by their own standard tracking error. As you can see, **being the most demanding method of the five (the PSO) does not grant the best performance**.

* The second table represents a summary of the portfolios broken down by single stocks. **Negative-weighted stocks are those that appear to be negatively correlated with the index and therefore should be bought short**. Indeed, all the weights in a portfolio always sum up to 100%.

* The last table shows the tracking error time series by optimization methods. **I decided to set up an interval of +/- 5 bps, within which the replica can be considered correct, to evaluate the mean of the tracking errors day by day**. I have also created a boolean variable to express this latter information.

Starting from this basis, I trained two models (the ML model and the DL model) to generate forecasts of the boolean variable, using the tracking errors of only 4 of the 5 portfolio optimization methods as predictors.

**I intentionally excluded PSO because it is the most expensive in terms of time and computational resources**. The ultimate goal is to compare models that can generate predictions based on the tracking errors of four optimization methods without the need for input from the fifth method (the PSO).

**If the models prove reliable, their predictions on a completely new dataset could be interpreted as if they were generated using information from PSO as well**. Even if, in reality, PSO was not actually applied to the new data, which would lead to a significant saving of resources.
prova