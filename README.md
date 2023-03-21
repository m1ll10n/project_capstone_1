![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
# COVID-19 Prediction with LSTM in Malaysia

This GitHub repository contains a deep learning model using LSTM neural network to predict the daily number of COVID-19 cases in Malaysia. The model uses the past 30 days of cases data to forecast the number of cases for the next day, providing a useful tool for COVID-19 forecasting.


## Applications
Below are the steps taken on solving the task.
### 1. Exploratory Data Analysis
1.1 Check the cases_new column and plotting it in line plot for time series problem.
![data inspect](https://user-images.githubusercontent.com/49486823/226570648-b8515947-6abe-4ad8-a68d-5c4e9902af54.png)


### 2. Data Cleaning
Data cleaning are done by:

2.1 Changing non-numeric values such as ' ' to null values with to_numeric.
```python
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')
```
2.2 Interpolating missing data rather than removing the rows as it is time series data.
```python
df['cases_new'] = df['cases_new'].interpolate(method='polynomial', order=2)
```
2.3 Selecting cases_new column as data for feature and target variable.
```python
data = df['cases_new'].values
```
The data is missing between 400 to 500 on x-axis.

Before Data Cleaning:

![before clean](https://user-images.githubusercontent.com/49486823/226570748-2364e98c-b755-4c4f-92bf-9abcf4d40a92.png)


After Data Cleaning:

![after clean](https://user-images.githubusercontent.com/49486823/226570808-29e83f19-0bdb-42ba-b9bf-a7da6cfe56dc.png)

### 3. Data Preprocessing
There are two(2) steps of data preprocessing:

3.1 Scale data with Min-Max Scaler.\
3.2 Window size is set to 30 for past 30 days of data as feature variable and target variable as a day after.

### 4. Model Development
This is the model architecture. Few notable settings not included in the screenshot:

4.1 LSTM neurons are set to 64 as limit to prevent from brute forcing the problem but number of layers aren't limited.\
4.2 Test data is 20% and the rest is allocated to training data.\
4.3 Batch size is 64 with 600 epochs.\
4.3 Metrics is Mean Absolute Error Percentage and Mean Squared Error.\
4.4 Optimizer is Adam Optimization.\
4.5 Loss function is Mean Absolute Error Function.\
4.6 No early stopping implemented.

![model](https://user-images.githubusercontent.com/49486823/226570980-90cd50ee-f316-4642-a976-efac08edbcfc.png)

## Results
This section shows all the performance of the model and the reports.
### Training Logs
The model shows training loss/mse is higher than validation loss/mse. But the training shows signs of fluctuation which might be a sign of overfitting.

Training Loss:

![Wan_Umar_Farid_Training_Loss](https://user-images.githubusercontent.com/49486823/226571084-5564257d-f3dd-4e32-97c8-1a62a86867cc.jpg)

Training MSE:

![Wan_Umar_Farid_Training_MSE](https://user-images.githubusercontent.com/49486823/226571098-9e65387c-a8bc-405b-ab10-c4c9ec563cf7.jpg)

### Model Performance
The goal of this problem statement is to achieve an Mean Absolute Percentage Error of lower than 10% on testing dataset which is 100 days after the training dataset. And the model achieved an MAPE of 8.17%.

Testing Performance:

![Wan_Umar_Farid_Test_Perf](https://user-images.githubusercontent.com/49486823/226571180-f80ca61d-a692-4dc1-af83-30f0bd957a05.jpg)

Predicted cases from the model against actual cases.

![Wan_Umar_Farid_Prediction_Graph](https://user-images.githubusercontent.com/49486823/226571215-6d3d8e91-c89c-4218-a245-c3ef57936291.png)

## Credits
Data can be obtained from https://github.com/MoH-Malaysia/covid19-public .
