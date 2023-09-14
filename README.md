# **Learning Portfolio Optimization Methods**

# 1. Introduction

The Dow Jones Industrial Average (“DJIA”, below) is a popular stock market index that brings together 30 major companies listed on U.S. stock exchanges. [Check here for general knowledge](https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average) and [here for the technical details](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-dj-averages.pdf).

The official DJIA close is calculated by summing all the close prices of the stocks comprised in the index and dividing by a constant, the Dow Divisor. The purpose of the Dow Divisor is to ensure that extraordinary events like stock splits, spin-offs, or other structural changes do not in them-selves alter the numerical value of the DJIA. Nevertheless, a committee of professionals adjusts the actual composition of DJIA to reflect the trajectory of the American economy regularly. This means that the composition is the result of a subjective decision. [For the historical components of the ^DJI check here](https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average).

I decided to base this work on the DJIA due to its vogue, but other market indexes would also work perfectly. Just like individual stocks, stock market indexes also produce a return (positive or negative) over time.

# 3. Building the Portfolios

To produce optimal portfolios, I used five well-known portfolio optimization algorithms using the DJIA's daily adjusted close price on Yahoo Finance in respect to its stock components. The time frame under consideration is from June 1, 2003 through May 31, 2023. To allow for splits and dividend payments, I utilized modified closing prices rather than ordinary closing prices. The methods employed are as follows.

*   Non-Negative Least Squares ('NNLS', below). The NNLS model is a sort of restricted least squares problem in which the coefficients cannot turn negative.
*   Partial Correlation ('PCRR', below). The PCRR model computes the degree of association between two random variables after adjusting for a collection of random factors.
*   Dynamic Time Warping ('DTW', below). The DTW model is a technique for determining the similarity of two temporal sequences with varying speeds.
*   Non-Negative Matrix Factorization ('NNMF', below). The NNMF model is a collection of multivariate analysis and linear algebra methodologies for factoring a matrix into (typically) two matrices with no negative components, each easier to inspect.
*   Particle Swarm ('PSO', below). The PSO model generates a population of potential solutions (particles) and moves them around the search space using a simple mathematical formula that takes the particle's position and velocity into consideration.

Thanks to these methods I obtained the weights for each stock (summing up to 1) to include in the portfolios. Negative-weighted stocks are meant to be bought short.

Every proper portfolio optimization strategy has periodical (monthly or trimestral) checkpoint buckets to evaluate the goodness of the replica and make some rebalancing. At every checkpoint, stock weights in the portfolio can change, so that some stocks can be excluded from the next bucket and replaced by others. For simplicity, I will not make periodical bucket rebalancing, and I have assumed that all the stocks of the DJIA should get a spot in the portfolios.

![img.png](img.png)

# 4. Training Phase

For each optimized portfolio, I calculated the tracking error ('TE', below) which is the absolute difference in real returns (in percentage points) between the portfolio replica and the DJIA index. The standard deviation of the discrepancies between the portfolio's and the DJIA's returns yields the TE. Because the purpose is to duplicate an index rather than outperform the market, I am only interested in portfolios with minimal TE in absolute terms.

The TE data was processed using machine learning and deep learning techniques. The final objective is to train a model capable of generalizing the optimization approaches' combined findings. To examine the mean of the tracking errors day by day, I arbitrarily set a +/- 5 bps interval within which the replica may be regarded correct. To express this subsequent knowledge, I also constructed a Boolean variable.

Beginning with this, I constructed two learning models to forecast the Boolean variable, using the tracking errors of each of the five optimization procedures as predictors. I used a Random Forest classifier ('RFC', below) for the machine learning model and a very simple Multi Layer Perceptron ('MLP', below), with structure 4-2-1, for the deep learning model to make things as simple as possible.

# 5. Results & Conclusions

As follows there are the accuracy scores of both the machine learning and deep learning models grouped by couple of predictors.

*   NNLS + PCRR: `ml accuracy: 0.8347, dl accuracy: 0.8858`
*   NNLS + DTW: `ml accuracy: 0.6523, dl accuracy: 0.6459`
*   NNLS + NNMF: `ml accuracy: 0.8540, dl accuracy: 0.8547`
*   NNLS + PSO: `ml accuracy: 0.8148, dl accuracy: 0.8149`
*   PCRR + DTW: `ml accuracy: 0.7725, dl accuracy: 0.7859`
*   PCRR + NNMF: `ml accuracy: 0.9197, dl accuracy: 0.9048`
*   PCRR + PSO: `ml accuracy: 0.8931, dl accuracy: 0.8832`
*   DTW + NNMF: `ml accuracy: 0.8260, dl accuracy: 0.7938`
*   DTW + PSO: `ml accuracy: 0.7275, dl accuracy: 0.7035`
*   NNMF + PSO: `ml accuracy: 0.8717, dl accuracy: 0.8837`

The PCRR and DTW portfolio optimization models yielded the best pair of predictors. It was able to generalize the information from the training set to more than 90% of the days in the test set.

Overall, the predictors created by combining two optimization strategies proved to be fairly trustworthy. Their predictions on a completely fresh data set might be viewed as if they were made using data from all approaches. Even if three of the five optimization strategies were not actually applied to the new data. This would result in huge resource savings.
