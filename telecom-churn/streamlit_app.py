import streamlit as st
import requests

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📊", layout="wide")


API_URL = "http://127.0.0.1:8000/predict"
HEALTH_URL = "http://127.0.0.1:8000/health"

st.title("📊 Telecom Customer Churn Prediction")

st.markdown("""
    Predict customer churn probability using the trained
    Logistic Regression model.
    """)

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=5
    )

    onlinesecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    techsupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


with col2:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperlessbilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    paymentmethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthlycharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=0.01
    )


st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    use_container_width=True
)


if predict_button:

    customer_data = {
        "tenure": tenure,
        "onlinesecurity": onlinesecurity,
        "techsupport": techsupport,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
    }

    try:

        with st.spinner("Connecting to prediction API..."):

            health_response = requests.get(
                HEALTH_URL,
                timeout=10
            )

        if health_response.status_code != 200:

            st.error(
                "❌ FastAPI service is not healthy."
            )

            st.stop()

        with st.spinner("Analyzing customer..."):

            response = requests.post(
                API_URL,
                json=customer_data,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            probability = result[
                "churn_probability"
            ]

            prediction = result[
                "prediction"
            ]

            churn = result[
                "churn"
            ]

            risk_level = result[
                "risk_level"
            ]

            st.success(
                "✅ Prediction completed successfully!"
            )

            st.header("Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with col2:

                st.metric(
                    "Prediction",
                    churn
                )

            with col3:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            st.subheader("Churn Risk")

            st.progress(
                min(float(probability), 1.0)
            )

            if risk_level == "HIGH":

                st.error(
                    """
                    ⚠️ **High Churn Risk**

                    This customer has a high probability
                    of leaving the company. Consider
                    prioritizing this customer for a
                    retention campaign.
                    """
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    """
                    ⚠️ **Medium Churn Risk**

                    This customer shows some signs of
                    churn risk. Consider monitoring the
                    customer and providing targeted offers.
                    """
                )

            else:

                st.info(
                    """
                    ✅ **Low Churn Risk**

                    This customer currently has a low
                    probability of churn.
                    """
                )

            st.subheader("💡 Business Recommendation")

            if risk_level == "HIGH":

                st.write(
                    """
                    - Prioritize this customer for retention.
                    - Consider a personalized offer.
                    - Review contract and pricing.
                    - Offer additional support services.
                    """
                )

            elif risk_level == "MEDIUM":

                st.write(
                    """
                    - Monitor customer activity.
                    - Consider targeted engagement.
                    - Promote useful additional services.
                    """
                )

            else:

                st.write(
                    """
                    - Continue normal customer engagement.
                    - Maintain service quality.
                    - No immediate retention action is required.
                    """
                )

            with st.expander(
                "🔍 Model Information"
            ):

                st.write(
                    f"**Model:** "
                    f"{result['model_name']}"
                )

                st.write(
                    f"**Version:** "
                    f"{result['model_version']}"
                )

                st.write(
                    f"**Prediction Class:** "
                    f"{prediction}"
                )

        else:

            st.error(
                f"❌ API Error: {response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            """
            ❌ Cannot connect to FastAPI backend.

            Make sure FastAPI is running:

            `uv run uvicorn main:app --reload`
            """
        )

    except requests.exceptions.Timeout:

        st.error(
            "❌ API request timed out."
        )

    except Exception as e:

        st.error(
            f"❌ Unexpected error: {str(e)}"
        )