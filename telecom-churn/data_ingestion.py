import pandas as pd


class DataIngestion:

    # All 17 columns required by the saved preprocessor
    INPUT_COLUMNS = [
        "partner",
        "dependents",
        "tenure",
        "multiplelines",
        "internetservice",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "contract",
        "paperlessbilling",
        "paymentmethod",
        "monthlycharges",
        "totalcharges",
        "seniorcitizen",
    ]

    # Features collected from the user (non-zero L1 coefficients only)
    USER_INPUT_COLUMNS = [
        "tenure",
        "onlinesecurity",
        "techsupport",
        "contract",
        "paperlessbilling",
        "paymentmethod",
        "monthlycharges",
    ]

    NUMERIC_COLUMNS = [
        "tenure",
        "monthlycharges",
        "totalcharges",
        "seniorcitizen",
    ]

    # Default values for zero-importance features that are still
    # required by the preprocessor but not collected from the user.
    ZERO_IMPORTANCE_DEFAULTS = {
        "partner": "No",
        "dependents": "No",
        "multiplelines": "No",
        "internetservice": "No",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "totalcharges": 0.0,
        "seniorcitizen": 0,
    }

    def validate_user_columns(self, customer_data):

        missing_columns = [
            column for column in self.USER_INPUT_COLUMNS
            if column not in customer_data
        ]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

    def fill_defaults(self, customer_data: dict) -> dict:
        """Fill in default values for zero-importance features."""
        full_data = dict(customer_data)
        for col, default in self.ZERO_IMPORTANCE_DEFAULTS.items():
            if col not in full_data:
                full_data[col] = default
        return full_data

    def create_dataframe(self, customer_data):

        self.validate_user_columns(customer_data)

        full_data = self.fill_defaults(customer_data)

        df = pd.DataFrame([full_data])
        df = df[self.INPUT_COLUMNS]

        return df

    def convert_numeric_columns(self, df):

        for column in self.NUMERIC_COLUMNS:

            df[column] = pd.to_numeric(df[column], errors="coerce")

        if df[self.NUMERIC_COLUMNS].isnull().any().any():

            raise ValueError("Invalid numeric value found in numeric columns.")

        return df

    def prepare_data(self, customer_data):

        df = self.create_dataframe(customer_data)

        df = self.convert_numeric_columns(df)

        return df