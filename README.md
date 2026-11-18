# Hotel-Booking-Cancellation-Predict
Predicting hotel booking cancellations using machine learning models and ensemble methods.

### Project Overview

This project was completed for my Capstone in Data Science course. The goal is to predict whether a hotel reservation will be canceled using machine learning. 

Hotel cancellations can create challenges for hotels, including lost revenue and difficulties with room availability. By identifying patterns associated cancellations, hotels may be able to make better decisions about reservations and planning. 

### Dataset

The project uses the Hotel Reservations dataset, which contains 36,275 hotel bookings and 19 original columns. 

The target variable is booking_status, which identifies whether a reservation was canceled or not canceled.

The dataset includes information such as:

- Lead time before arrival

- Average room price

- Number of adults and children

- Number of weekend and weekday nights

- Market segment

- Previous cancellations

- Special requests

- Parking requirements

### Data Preparation

The dataset was examined for missing values and duplicate records.

The Booking_ID column was removed because it was an identifier rather than a useful prediction feature. 

Two additional features were created:

- **total_nights:** the combined number of weekend and weekday nights
- **total_guests:** the combined number of adults and children

Categorical variables were converted into numerical features using one-hot encoding. Numerical features were scaled for models that required scaling. 

The dataset was divided into 80% training data and 20% testing data. 

### Machine Learning Models

Five machine learning models were trained and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest
5. XGBoost

Random Forest and XGBoost were also tune using hyperparameter tuning to improve their predictive performance. 

### Model Performance

The models were assess using accurarcy, precision, recall, and F1-score. 

'''text
| Model | Accuracy | F1-Score |
| :--- | ---: | ---: |
Logistic Regression | 81.43% |69.45% |
KNN | 85.17% | 76.65% |
Decision Tree | 87.47% | 81.00% |
Random Forest | 90.35% | 84.62% |
XGBoost | 89.91% | 84.07% |
Tuned Random Forest | 90.41% | 84.71% |
Tuned XGBoost | 90.41% | 84.93% |

#### Tuned XGBoost achieved the highest F1-score of 84.93% while both tune models achieved 90,41% accuracy. 

### Key Findings

The exploratory analysis found several differences between canceled and non-canceled reservations.

- Canceled reservations had longer average lead times.
- Canceled reservations had higher average room prices.
- Reservations with more special request were less likely to be canceled in the observed data.

The tuned XGBoost model identified online market segment, parking requirements, special requests, and lead time among its important prediction features. 

These findings describe patterns in the dataset and do not establish that these features cause cancellations

### Tools and Technologies

- Python
- Google Colab
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Github

### How to Run the Project

1. Open the Jupyter notebook (.ipynb) in this repository using Google Colab.
2. Run the notebook cells in order.
3. The notebook loads the Hotel Reservations CSV dataset directly from this public GitHub repository.
4. Review the exploratory analysis, machine learning models performance comparisons, and visualizations.

### Conclusion

The project demonstrates how machines learning can be used to predict hotel booking cancellations. 

The results show that ensemble learning methods, particularly Random Forest and XGBoost, performed well on this dataset.

The tuned XGBoost model achieved the highest F1-score making it the preferred model when balancing precision and recall for cancellation prediction. 
