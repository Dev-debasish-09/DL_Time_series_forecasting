# <div align="center">Time series forecasting using Deep Learning</div>
<div align="center"><img src="https://github.com/Dev-debasish-09/DL_Time_series_forecasting/blob/main/output/overview.gif?raw=true"></div>



## Overview:
Deep learning methods offer a lot of promise for time series forecasting, such as the automatic learning of temporal dependence and the automatic handling of temporal structures like trends and seasonality.

## Dataset:
[Predict Future Slaes](https://www.kaggle.com/c/competitive-data-science-predict-future-sales)<br>
[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/data)
#### Data Fields:
- **ID** - an Id that represents a (Shop, Item) tuple within the test set
- **shop_id** - unique identifier of a shop
- **item_id** - unique identifier of a product
- **item_category_id** - unique identifier of item category
- **item_cnt_day** - number of products sold. You are predicting a monthly amount of this measure
- **item_price** - current price of an item
- **date** - date in format dd/mm/yyyy
- **date_block_num** - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
- **item_name** - name of item
- **shop_name** - name of shop
- **item_category_name** - name of item category
- **store** - Store ID
- **sales** - Number of items sold at a particular store on a particular date.

#### Time period of dataset:
```
Min date from train set: 2013-01-01
Max date from train set: 2017-12-31
```

## Implementation:

**Libraries:**  `NumPy` `pandas` `tensorflow` `matplotlib` `sklearn` `seaborn`
## Data Exploration:
<img src="https://github.com/Dev-debasish-09/DL_Time_series_forecasting/blob/main/output/overall%20daily%20sales.PNG?raw=true">
<img src="https://github.com/Dev-debasish-09/DL_Time_series_forecasting/blob/main/output/item%20daily%20sales.PNG?raw=true">
<img src ="https://github.com/Dev-debasish-09/DL_Time_series_forecasting/blob/main/output/store%20sales.PNG?raw=true">

## Model training, evaluation, and prediction:
### Multilayer Perceptron:
- Multilayer Perceptron model or MLP model, here our model will have input features equal to the window size.
- MLP models don't take the input as sequenced data, so for the model, it is just receiving inputs and don't treat them as sequenced data, that may be a problem since the model won't see the data with the sequence pattern that it has.

### CNN Model:
- For the CNN model we will use one convolutional hidden layer followed by a max pooling layer. The filter maps are then flattened before being interpreted by a Dense layer and outputting a prediction.
- The convolutional layer should be able to identify patterns between the timesteps.

### LSTM:
- Now the LSTM model actually sees the input data as a sequence, so it's able to learn patterns from sequenced data (assuming it exists) better than the other ones, especially patterns from long sequences.

### CNN-LSTM:
- CNN-LSTM is a hybrid model for univariate time series forecasting.

- The benefit of this model is that the model can support very long input sequences that can be read as blocks or subsequences by the CNN model, then pieced together by the LSTM model.

### Comapring Models:
<img src ="https://github.com/Dev-debasish-09/DL_Time_series_forecasting/blob/main/output/compare%20models.PNG?raw=true">
<br>

| Model             | Train RMSE             | Validation RMSE                                                                |
| ----------------- | -----------------| ------------------------------------------------------------------ |
| MLP | 18.36|  18.50 |
| CNN | 18.62|  18.76 |
| LSTM | 19.98|  18.76 |
| CNN-LSTM| 19.20 |  19.17 |
### Lessons Learned
`Time Series Forecasting`
`Deep Learning for Time Series Forecasting`
`LSTM`








## References:
[Deep Learning for Time Series Forecasting](https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/)
### Feedback

If you have any feedback, please reach out at debasishpradhan1934@gmail.com

## 👨‍💻 Author
**Debasish Pradhan**  
AI Enthusiast | Data Science & Machine Learning Practitioner  
📧 Email: debasishpradhan1934@gmail.com  
🔗 GitHub: https://github.com/Dev-debasish-09






