# %% Imports
# Data manipulation
import pandas as pd
import numpy as np

# EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from lightgbm import LGBMClassifier

# Pre-processing
import optuna
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from category_encoders import TargetEncoder, OrdinalEncoder
# %% Load and Prepare Data
df = pd.read_csv("../data/raw/Hotel Reservations.csv")
df = df.drop(columns = ['Booking_ID', 'arrival_year'], axis = 1).copy()
df['booking_status'] = df['booking_status'].map({'Canceled': 1, 'Not_Canceled': 0})

features = df.drop(columns = 'booking_status', axis = 1).columns.to_list()
target = 'booking_status'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=21)
# %% Preprocessing pipelines
cat_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'repeated_guest', 'required_car_parking_space']
num_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'lead_time', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']
ordinal_features = ['arrival_date', 'arrival_month']

cat_transformer = Pipeline([
    ('imput_cat', CategoricalImputer(imputation_method='frequent')),
    ('encoder_cat', TargetEncoder())
])

num_transformer = Pipeline([
    ('imput_num', MeanMedianImputer(imputation_method='median'))
])

ordinal_transformer = Pipeline([
    ('imput_or', MeanMedianImputer(imputation_method='median')),
    ('encoder_or', OrdinalEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features),
        ('ordinal', ordinal_transformer, ordinal_features)
    ]
)
# %% Model training
params_dict = {'learning_rate': 0.06870145636595233,
               'num_leaves': 997,
               'subsample': 0.7862997832574106,
               'colsample_bytree': 0.9347307739548064,
               'min_data_in_leaf': 2}

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(**params_dict, 
                                  n_estimators = 1000, 
                                  verbosity = -1, 
                                  random_state=21))
])

clf.fit(X_train, y_train)
# %% Model evaluation
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

model_metrics = {
    'Accuracy': metrics.accuracy_score(y_test, y_pred),
    'F1 Score': metrics.f1_score(y_test, y_pred),
    'ROC AUC': metrics.roc_auc_score(y_test, y_proba)
}

print(model_metrics)
# %% Save model
model_series = pd.Series({
    'model': clf,
    'features': features,
    'metrics': model_metrics
})

model_series.to_pickle("../models/classifier.pkl")