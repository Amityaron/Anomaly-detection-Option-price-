# Anomaly-detection-Option-price-
This application will change your way you see the stock market.

[https://anomaly-detection-option-price.streamlit.app/](https://main-app1.streamlit.app/)

This application will transform the way you perceive the stock market. 

By utilizing statistical measures such as mean, mode, interquartile range, standard deviation, and z-score, along with the Black-Scholes model, you can identify opportunities that were previously unseen.

The application consists of three parts:

1.An anomaly detection tools for the stock market using mean, mode, interquartile range, and standard deviation.
2. A strategy for selling call options based on the Black-Scholes model.
4. A Z-score analysis with skewness and kurtosis to identify potential long positions.


## anomaly detection tools:

Based on mean and standard deviation bollinger band method idetify local minition in the graph.

**Exmaple of the tiker QQQ from 1/1/23 until 29/8/24:**

<img src="https://github.com/user-attachments/assets/d9431f91-440d-465c-a2da-81880484d249" width="500" height="500">

The image below calculates the gain percentage, assuming you are using a buy-and-hold strategy with only long signals.

<img src="https://github.com/user-attachments/assets/e25eb8f3-956a-4d33-a871-9c812c0b59df" width="500" height="400">

The image below presents the interquartile range percentage change for each month.
Statistically, each year should fall within the interquartile range. 
If there is a year where the red points consistently fall outside the interquartile range month after month, it signals an outlier year.

<img src="https://github.com/user-attachments/assets/91190967-792f-4bc0-bb52-ff5421957eda" width="500" height="500">

Finally, the image below presents the probability of achieving a positive return each month.
We focus on the months with the lowest chance of a positive return to accumulate long positions at lower prices.

<img src="https://github.com/user-attachments/assets/34e1cc91-d461-47b4-8329-b41481ba6e75" width="500" height="500">


## Sell call options strategy

Sell call options strategy to get mouthly income.

Based on your index funds you have you can sell call option with the minimum probability of expiring OTM(OUT OF THE MONEY)

![image](https://github.com/user-attachments/assets/87b30b51-3c2a-4c33-af89-485d1e51d762)

## anomaly detection based on z score 


Identify the histograms shapes of different index funds with skewness and kurtosis and determine local minimums based on a Z-score of -2 or lower.


![image](https://github.com/user-attachments/assets/6ad0c16b-4dd8-4314-b40f-567c9164023b)


![image](https://github.com/user-attachments/assets/cf37f671-7a07-4c49-bba1-79780c3764a1)











