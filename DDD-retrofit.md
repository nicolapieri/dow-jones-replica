# **Analysis** - Glossary

**DJIA index**
*   stands for dow jones industrial average
*   is composed by 30 publicly traded stocks
*   has its own daily closing price

**DJIA composition**
*   represents the list of closing price time series of the stocks that compose the DJIA index in a given time period
*   is defined within a specific start date and end date
*   can change over time

**portfolios**
*   represent tentative replicas of the DJIA index daily closing price
*   are built applying to the DJIA composition each of the optimization methods NNLS, PCRR, DTW, NNMF and PSO
*   deviate from the actual DJIA index by a tracking error on a daily basis

**predictors**
*   represent the two selected optimization methods whose 2 tracking errors will be used to generalize the information of all the 5 tracking errors
*   are chosen before every learning session
*   must be two and must be different from each other

**learning models**
*   are represented by one machine learning and one deep learning algorithms with fixed structure and hyperparameters
*   learn whenever the mean of the 5 tracking errors exceeds a predefined threshold on a daily basis
*   use 2 tracking error time series as predictors

# **Analysis** - Use Cases

*   **Sub-domain - DJIA monitoring (supporting)**
  *    **Use case - DJIA closing prices acquisition:**
      1.   system connects to yahoo finance
      2.   system downloads the last 20 years DJIA closing price time series

  *    **Use case - DJIA composition update:**
      1.   system crawls DJIA composition from wikipedia (https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average)
      2.   system updates the DJIA composition
      3.   system connects to yahoo finance
      4.   system downloads the last 20 years stocks closing price time series belonging to the DJIA composition

*   **Sub-domain - portfolio optimization (generic)**
  *    **Use case - replica portfolios generation:**
      1.   system applies the 5 optimization methods using the latest data available
      2.   system builds up 5 portfolios corresponding to the 5 optimization methods

  *    **Use case - determining tracking error threshold:**
      1.   system computes portfolios' tracking errors and their mean
      2.   system defines a threshold of +/- 0.05% around the mean tracking error and maps a related boolean variable
      3.   system show a chart with the trend of tracking errors and the threshold

*   **Sub-domain - learning session (core)**
  *    **Use case - predictors choice:**
      1.   system asks to choose two different optimization methods among the 5 available to use as predictors
      2.   system checks if the choices are valid

  *    **Use case - models training and testing:**
      1.   system trains a machine learning model and a deep learning model using the 2 chosen predictors
      2.   system tests the models and reports the accuracy

# **Design** - Bounded Contexts description

*   **DJIA index context**
    * this bounded context concerns the sub-domains of DJIA index monitoring
    * examples of functionalities:
      * gathering external data
      * updating internal data
      * defining the functions for all the steps of the pipeline

*   **Portfolios context**
    * this bounded context concerns the sub-domain of portfolio optimization
    * examples of functionalities:
      * applying optimization methods
      * computing portfolios tracking errors
      * defining replica threshold

*   **Learning context**
    * this bounded context concerns the sub-domain of learning session
    * examples of functionalities
      * allowing the choice of predictors
      * training the learning models
      * testing the models

# **Design** - Software Architecture description

*   **Functions microservice**
    *   this microservice concerns the abilitation of all the functions needed by the subsequent phases of the pipeline (except for the learning session to avoid circular import).

*   **Processing microservice**
    *   this microservice performs the first steps of the pipeline:
      * DJIA data ingestion
      * Portfolio generation
      * Computing tracking errors
      * Setting threshold

*   **Learning microservice**
    *   this microservice performs the lasts steps of the pipeline:
      * Choosing the predictors
      * Splitting dataset
      * Training models
      * Testing & accuracy
