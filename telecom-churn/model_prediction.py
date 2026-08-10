import joblib

from shap_explainer import ChurnExplainer


class ChurnPredictor:

    def __init__(self, model_path: str):

        self.model_path = model_path
        self.artifact = joblib.load(self.model_path)

        self.model = self.artifact["model"]

        self.preprocessor = self.artifact["preprocessor"]

        self.shap_background = self.artifact["shap_background"]

        self.threshold = self.artifact.get("threshold", 0.5)

        self.model_name = self.artifact.get("model_name", "Logistic Regression")

        self.model_version = self.artifact.get("model_version","1.0")

        self.explainer = ChurnExplainer(
            model=self.model,
            preprocessor=self.preprocessor,
            background_data=self.shap_background)

    def preprocess(self, customer_df):

        X_processed = self.preprocessor.transform(customer_df)

        return X_processed

    def predict_probability(self, X_processed):

        probability = self.model.predict_proba(X_processed)[0, 1]

        return float(probability)

    def predict_class(self, probability):

        return int(probability >= self.threshold)

   
    def get_risk_level(self, probability):

        if probability >= 0.70:
            return "HIGH"

        elif probability >= 0.40:
            return "MEDIUM"

        return "LOW"

    def predict(self, customer_df):

        X_processed = self.preprocess(customer_df)

        probability = self.predict_probability(X_processed)

        prediction = self.predict_class(probability)

        risk_level = self.get_risk_level(probability)

        top_features = self.explainer.get_top_features(customer_df, top_n=5)

        return {"churn_probability": round(
                probability, 4),
                "prediction": prediction,
                "churn": ("Yes" if prediction == 1 else "No"),
                "risk_level": risk_level,
                "top_churn_factors": top_features,
                "model_name": self.model_name,
                "model_version": self.model_version}