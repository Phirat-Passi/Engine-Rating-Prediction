# Engine Rating Prediction 

This project features a Python script that predicts engine ratings based on inspection parameters 
exclusively derived from the provided dataset, contributing valuable insights to the automotive industry.

# Objective

Your task is to write a small Python script that predicts the engine rating based on the inspection 
parameters using only the provided dataset. You need to find all the cases/outliers where the rating has been 
given incorrectly as compared to current condition of the engine.
This task is designed to test your Python ability, your knowledge of Data Science techniques, 
your ability to find trends, outliers, relative importance of variables with deviation in target variable 
and your ability to work effectively, efficiently and independently within a commercial setting.
This task is designed as well to test your hyper-tuning abilities or lateral thinking.

## Data

![Screenshot from 2024-01-22 17-13-45](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/04cb36fe-8d94-47f4-bc9e-64bf9965a291)

## Null Value in Data

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/c9cdc0f9-0540-46f2-bc02-0c0d8f8e79cd)

## Variable in Data 

![Screenshot from 2024-01-23 10-24-24](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/bdfed7fe-bb3f-46ed-ac45-abe244fa07ae)

## EDA of Data

![Screenshot from 2024-01-23 10-26-23](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/1a900ecb-b3c4-4beb-9fb2-8f1c167b5f63)

# Univariate Analysis 

## Number of Inspections per date

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/c0a340c9-7290-4b29-8780-b2afaade1cc5)

## Number of Insections per month

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/dc5e4c82-82af-4934-8d76-25e3335ec639)

## Visualizing Number of Inspections across day of month

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/58340ee0-97da-4760-b955-fd6b9ae8f6b4)

## Number of INspections across weekday 

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/0719556c-9ec4-4bd5-a0eb-03ba55d5ebf0)

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/fa3b6e02-e3b4-4cbb-aef4-4d836ee01774)

Analysis of Registration Year

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/cba79a73-9c8f-4e85-94f9-f9cfdff6bbdc)

# Bivariate Analysis 

## Inspection Month v/s Rating Engine Transmission

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/eb221091-cbe1-435a-a428-b40ae6bbb031)

## Correlation Matrix 

![image](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/574a9b69-6c23-4864-a513-88f09d454b34)

# Model Training & Testing 

![Screenshot from 2024-01-23 10-36-05](https://github.com/Phirat-Passi/Engine-Rating-Prediction/assets/67471222/74853876-1d6c-49c4-b58f-5ff78df3cf05)

# Please answer the following:

#### 1. Briefly describe your approach to this problem and the steps you took

#### Ans:    My Approach

1. Import all the necessary libraries/packages 

2. Load the dataset into memory
3. EDA of data
      1. data.shape : concludes there are a total 26307 rows & 73 columns (including target variable) - 72 features
      2. data.info : gives data type of each feature.
            - here type of inspectionStartTime column is object and it is date type column so we need to change its data type to date and time
      3. data.describe - provides basic statistical inference of each column (max value, min value, mean, median, mode)
      4. Finding & visualising NaN values (sum & percentage of NaN values in each column)
            1. Total number of columns which has more than 40% null values in it: 52 
                  - (52 out of 72 independent varaibles have more then 40% null values)
                  
            2. Hypothesis : based on fields summary it seems that " NaN values need to be imputed as Yes in most columns "
                  - hypothesis testing - by Intersection check where,
                  - Set-A : column with their description as "current condition if not yes"
                  - set-B : col with more than 40% NaN values
                  - observation: set A intersection set B resulted in a Non-empty set, proving that the hypothesis of (NaN values should be imputed as yes) is True.
            3. intersection = 47 / 52 
                  - (47 out of 52 variables have a description of "current condition if not yes" -> these are imputed as yes)
            4. lets check other 5 columns that have null values but condtion is other then current "condition if not yes" 
                  - As we can analyze that these columns have more the 80% of null values and they are also not under condition of 40% more null values in it.
                  - We drop those colums as they are not important as for our analysis
                        
      5. Checking for categorical variables & continuous variables
            1. Total Categorcal data in DB :  63
            2. total continuous data in DB including target variable :  5

      6. Univariate Analysis
            1. inspectionStartTime: 
                  - The inspection data is from 2nd January 2019 to 15th April 2019.
                  - Extract the inspection month, day, dayofweek & hour
                  - Average daily inspections (for this sample): 257.9117647058824.
                  - We have inspection data for 102 dates.
                  - month-wise : monthly representation is maximum for March (~ 30%) followed by January (~ 27.4%), February(~ 27%) & April (~ 15.4%).
                  - day-wise : Few high spikes of inspections can be seen on 3rd, 10th, & 14th of the month.
                  - week-wise : Average Weekly Inspections: 3758.1428571428573
                              Average inspections on Weekends: 4963.0
                  - Clearly, there are more inspections on Weekends. (Average inspections on weekend is greather than weekly average inspeactions by 32%.)
                  - hour-wise: There's a spike in number of inspections across from 12 am to 4 pm.
            2. Year : registeration year
                  - registeration year data ranges from 1989 to 2019 (max registration year = 2019 and min registration year = 1989)
                  - create a monthly mappingOdometer has right skewed observation with a lot of outliers.
                  - outlier removal using IQR (Inter quartile range) technique
                  - max car were registered in January-> 5132 & min in December ->1371
                  - max car reg in 2012 -> 2922
                  - min car reg in 2019 -> 10
            3. odometer rating
                  - Odometer reading is range from 1 to 999999
                  - Odometer has right skewed observation with a lot of outliers.
                  - oulier removal using IQR method
                  - Max Car odometer_reading 100000  :  8
                  - Min Car odometer_reading :  1
            4. rating_engineTransmission (Target variable)
                  - rating ranges from 0.5 to 5.0 (having 10 values)
                  - rating_engineTransmission 0.5, 1.5 & 2.0 have a very less count. (less than 1 %)
                  - avg rating is 3.62

      7. Analysis of categorical data
            - majority of categorical data have very hgh frequency

      8. Bivariate Analysis (checking Relation b/w features)
            - there is a dependence of inspectionMonth on rating_engineTransmission.
            - Plot a correlation matrix
            - year (registration year) & odometer_reading has significant negative correlation with the rating_engineTransmission.

4. Data Preprocessing
      1) Split into training set & target
      2) encode the categorical variables (using get_dummies)
      3) scaling the data(using min max scaler)

5. Data Split (train & test split) -> 80:20

6. Model Training & Testing

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Train Scores</th>
      <th>Test Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LinearRegression</th>
      <td>0.425029</td>
      <td>-5.041663e+20</td>
    </tr>
    <tr>
      <th>KNN</th>
      <td>0.733909</td>
      <td>6.166457e-01</td>
    </tr>
    <tr>
      <th>DecisionTree</th>
      <td>1.000000</td>
      <td>4.039494e-01</td>
    </tr>
    <tr>
      <th>RandomForest</th>
      <td>0.959380</td>
      <td>7.069791e-01</td>
    </tr>
    <tr>
      <th>ExtraTree</th>
      <td>1.000000</td>
      <td>7.053729e-01</td>
    </tr>
    <tr>
      <th>XGBoost</th>
      <td>0.820358</td>
      <td>7.102714e-01</td>
    </tr>
    <tr>
      <th>LGBM</th>
      <td>0.766533</td>
      <td>7.199465e-01</td>
    </tr>
    <tr>
      <th>ANN-MLP</th>
      <td>0.808071</td>
      <td>6.254377e-01</td>
    </tr>
    <tr>
      <th>SupportVector</th>
      <td>-0.033129</td>
      <td>-4.480475e-02</td>
    </tr>
  </tbody>
</table>
</div>
     
- here we can see that the R squared score of all models that we listed above are 
      1. LinearRegression, KNN, DecisionTree, RandomForest, ExtraTrees, MLPRegressor, SupportVectorRegressor models are giving results with overfitting.
      2. XGboost, LGBM models which are showing generalization. (Testing R2 Score > 0.71)
- hence we will proceed with XGboost, LGBM for further finetuning.
      
6. Finetuning:
      1. LGBM : using RandomSearchCV
            - Training R squared score: 0.7969750516972294
            - Validation R squared score: 0.7211616261543252

      2. XGBoost 
            1. Grid search 
                  1. Best Hyperparameters: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 5, 'n_estimators': 200, 'subsample': 0.8}
                  2. Best Model Training R-squared: 0.8312914728546122
                  3. Best Model Validation R-squared: 0.7237182684128904

            2. Random search
                  1. Best Hyperparameters: {'subsample': 0.79, 'n_estimators': 400, 'min_child_weight': 1, 'max_depth': 7, 'learning_rate': 0.06, 'colsample_bytree': 0.6}
                  2. Best Model Training R-squared: 0.8507528591287232
                  3. Best Model Validation R-squared: 0.7260110945221425
                  
      3. Final Model: we will go with XGBoost as our final model because it shows more generealization to new/unseen data and slightly better test R2 score

7. Storing the final model in a pickel file
















