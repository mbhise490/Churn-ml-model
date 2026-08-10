import pandas as pd


class DataIngestion:

    INPUT_COLUMNS = [
        "seniorcitizen",
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
        "totalcharges"
    ]

    NUMERIC_COLUMNS = [
        "seniorcitizen",
        "tenure",
        "monthlycharges",
        "totalcharges"
    ]

    def validate_columns(self, customer_data):

        missing_columns = [column for column in self.INPUT_COLUMNS if column not in customer_data]

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

    def create_dataframe(self, customer_data):

        self.validate_columns(customer_data)

        df = pd.DataFrame([customer_data])
        df = df[self.INPUT_COLUMNS]

        return df

    def convert_numeric_columns(self, df):

        for column in self.NUMERIC_COLUMNS:

            df[column] = pd.to_numeric(df[column],errors="coerce")

        if df[self.NUMERIC_COLUMNS].isnull().any().any():

            raise ValueError("Invalid numeric value found in numeric columns.")

        return df

    def prepare_data(self, customer_data):

        df = self.create_dataframe(customer_data)

        df = self.convert_numeric_columns(df)

        return df