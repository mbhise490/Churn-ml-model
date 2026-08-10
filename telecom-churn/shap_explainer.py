import shap
import pandas as pd
import numpy as np


class ChurnExplainer:

    def __init__(self, model, preprocessor, background_data):
        self.model = model
        self.preprocessor = preprocessor
        self.background_data = background_data

        self.feature_names = (self.preprocessor.get_feature_names_out())

        self.explainer = shap.LinearExplainer(self.model, self.background_data)

    def preprocess(self, customer_df):
        X_processed = self.preprocessor.transform(customer_df)
        return X_processed

    def calculate_shap(self, customer_df):

        X_processed = self.preprocess(customer_df)

        shap_values = self.explainer(X_processed)

        return shap_values

    def get_top_features(self,customer_df,top_n=5):

        shap_values = self.calculate_shap(customer_df)

        values = np.asarray(shap_values.values)

        values = values[0]

        if values.ndim > 1:
            values = values[:, -1]

        feature_data = pd.DataFrame({
            "feature": self.feature_names,
            "shap_value": values,
            "abs_shap_value": np.abs(values)})

        feature_data = feature_data.sort_values(by="abs_shap_value", ascending=False)

        top_features = feature_data.head(top_n)

        return top_features.to_dict(orient="records")