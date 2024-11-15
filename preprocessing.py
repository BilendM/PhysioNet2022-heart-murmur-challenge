import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import random
import joblib


class DataPreprocessor:
    def __init__(self, csv_file_path):
        self.data = pd.read_csv(csv_file_path, on_bad_lines='skip')
        self.drop_unnecessary_features()
        self.drop_unknown_murmur()
        self.encode_sex_and_pregnancy_status()

    def drop_unnecessary_features(self):
        drop_features = ["Murmur locations", "Most audible location", "Systolic murmur timing",
                         "Systolic murmur shape", "Systolic murmur grading", "Systolic murmur pitch",
                         "Systolic murmur quality", "Diastolic murmur timing", "Diastolic murmur shape",
                         "Diastolic murmur grading", "Diastolic murmur pitch", "Diastolic murmur quality",
                         "Additional ID", "Outcome", "Campaign"]
        self.data.drop(drop_features, axis=1, inplace=True)

    def drop_unknown_murmur(self):
        self.data = self.data.drop(
            self.data[self.data['Murmur'] == 'Unknown'].index)

    def encode_sex_and_pregnancy_status(self):
        self.data["Sex"] = self.data["Sex"].map({"Female": 0, "Male": 1})
        self.data["Pregnancy status"] = self.data["Pregnancy status"].map({
            False: 0, True: 1})

    def normalise_ages(self):
        default_age_numbers = [0.5, 6., 72., 180., 240.]
        for i, row in self.data.iterrows():
            dist = default_age_numbers
            if row["Age"] not in default_age_numbers:
                dist = [abs(x - row["Age"]) for x in dist]
                minimum = dist.index(min(dist))
                self.data.at[i, 'Age'] = default_age_numbers[minimum]

    def handle_missing_height(self, pregnant):
        for index, row in self.data.iterrows():
            mask = row['Pregnancy status'] == pregnant
            if mask and (pd.isnull(row['Height']) or pd.isnull(row['Weight'])):
                if pd.notnull(row['Age']):
                    if pregnant:
                        similar_mask = (self.data['Sex'] == row['Sex']) & (
                            self.data['Age'] == "Adolescent") & (self.data.index != index)
                    else:
                        similar_mask = (self.data['Sex'] == row['Sex']) & (
                            self.data['Age'] == row['Age']) & (self.data.index != index)

                    height_replacement = self.data.loc[similar_mask, 'Height'].mean(
                    ) if pd.isnull(row['Height']) else row['Height']
                    weight_replacement = self.data.loc[similar_mask, 'Weight'].mean(
                    ) if pd.isnull(row['Weight']) else row['Weight']

                    random_add = random.randint(0, 20) if pregnant else 0

                    self.data.at[index,
                                 'Height'] = height_replacement
                    # self.data.at[index,
                    #              'Weight'] = weight_replacement + random_add

    def handle_missing_weight(self):
        # Filter rows with non-missing height and weight
        data_with_values = self.data.dropna(subset=['Height', 'Weight'])

        # Prepare the features (X) and target (y)
        X = data_with_values[['Height']]
        y = data_with_values['Weight']

        # Define the degree of the polynomial
        degree = 4  # This can be adjusted based on the relationship you observe

        # Create a polynomial regression model
        polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())

        # Fit the model
        polyreg.fit(X, y)

        # Predict and impute missing weights in the original DataFrame
        missing_weight = self.data['Weight'].isnull(
        ) & self.data['Height'].notnull()
        predicted_weights = polyreg.predict(
            self.data.loc[missing_weight, ['Height']])

        # Optionally, round predicted weights to 2 decimal places if needed
        predicted_weights = np.round(predicted_weights, 2)

        # Impute missing weights with the predictions
        self.data.loc[missing_weight, 'Weight'] = predicted_weights

        poly_model_path = 'regression_model.joblib'
        joblib.dump(polyreg, poly_model_path)

    def impute_missing_weight_with_loaded_model(self, model_path):
        # Load the model
        loaded_polyreg_model = joblib.load(model_path)

        # Predict and impute missing weights
        missing_weight = self.data['Weight'].isnull(
        ) & self.data['Height'].notnull()
        predicted_weights = loaded_polyreg_model.predict(
            self.data.loc[missing_weight, ['Height']])

        # Optionally, round predicted weights to 2 decimal places if needed
        predicted_weights = np.round(predicted_weights, 2)

        # Impute missing weights with the predictions
        self.data.loc[missing_weight, 'Weight'] = predicted_weights

    def replace_pregnant_ages(self):
        mask = self.data["Pregnancy status"] == 1
        random_states = np.random.rand(mask.sum())
        self.data.loc[mask, "Age"] = np.where(
            random_states > 0.33, "Adolescent", "Young Adult")

    def fill_null_ages(self):
        temp = self.data.copy()
        temp['Height_normalized'] = (
            temp['Height'] - temp['Height'].mean()) / temp['Height'].std()
        temp['Weight_normalized'] = (
            temp['Weight'] - temp['Weight'].mean()) / temp['Weight'].std()

        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters)
        temp['Cluster'] = kmeans.fit_predict(
            temp[['Height_normalized', 'Weight_normalized']])
        temp['Age'] = pd.to_numeric(temp['Age'], errors='coerce')
        cluster_ages = temp.groupby('Cluster')['Age'].mean().to_dict()
        self.data['Age'] = temp.apply(lambda row: cluster_ages[row['Cluster']] if pd.isnull(
            row['Age']) else row['Age'], axis=1)

    def age_float_conversion(self):
        age_conversion = {"Neonate": 0.5, "Infant": 6,
                          "Child": 72, "Adolescent": 180, "Young Adult": 240}
        self.data['Age'] = self.data['Age'].replace(age_conversion)

    def encode_murmur_and_outcome(self):
        self.data["Murmur"] = self.data["Murmur"].map(
            {"Present": 0, "Absent": 1})
        # self.data["Outcome"] = self.data["Outcome"].map(
        #     {"Abnormal": 0, "Normal": 1})

    def fill_null(self):
        self.data.fillna(value=0.0, inplace=True)

    def preprocess(self):
        self.handle_missing_height(False)
        self.replace_pregnant_ages()
        self.handle_missing_height(True)
        self.handle_missing_weight()
        self.age_float_conversion()
        self.fill_null_ages()
        self.normalise_ages()
        self.encode_murmur_and_outcome()
        return self.data

    def train_val_preprocess(self):
        self.handle_missing_height(False)
        self.replace_pregnant_ages()
        self.handle_missing_height(True)
        self.impute_missing_weight_with_loaded_model(
            'regression_model.joblib')
        self.age_float_conversion()
        self.fill_null()
        self.encode_murmur_and_outcome()
        return self.data
