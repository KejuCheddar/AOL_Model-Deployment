"""
streamlit_app.py — UI untuk memanggil SageMaker endpoint model Credit Score.

Jalankan dengan:
    streamlit run streamlit_app.py

Pastikan environment tempat Streamlit dijalankan sudah punya AWS credentials
yang valid (aws configure / IAM role) dengan izin sagemaker:InvokeEndpoint.
"""

import io
import json

import boto3
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Score Predictor", page_icon="💳", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — konfigurasi koneksi ke endpoint
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Konfigurasi Endpoint")
region = st.sidebar.text_input("AWS Region", value="us-east-1")
endpoint_name = st.sidebar.text_input("SageMaker Endpoint Name", value="credit-score-endpoint")
st.sidebar.caption("Pastikan kredensial AWS (aws configure / IAM role) sudah tersedia di environment ini.")


@st.cache_resource(show_spinner=False)
def get_runtime_client(region_name: str):
    return boto3.client("sagemaker-runtime", region_name=region_name)


def invoke_endpoint_json(payload: dict, region_name: str, ep_name: str):
    client = get_runtime_client(region_name)
    response = client.invoke_endpoint(
        EndpointName=ep_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


def invoke_endpoint_csv(csv_text: str, region_name: str, ep_name: str):
    client = get_runtime_client(region_name)
    response = client.invoke_endpoint(
        EndpointName=ep_name,
        ContentType="text/csv",
        Accept="application/json",
        Body=csv_text.encode("utf-8"),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.title("💳 Credit Score Prediction")
st.write("Aplikasi ini memanggil SageMaker endpoint untuk memprediksi kategori Credit Score (**Poor / Standard / Good**).")

tab_single, tab_batch = st.tabs(["🧍 Prediksi Satu Data (Form)", "📄 Prediksi Batch (Upload CSV)"])

# ---------------------------------------------------------------------------
# TAB 1 — Form input manual untuk satu nasabah
# ---------------------------------------------------------------------------
with tab_single:
    st.subheader("Masukkan Data Nasabah")

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Profil**")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            occupation = st.text_input("Occupation", value="Engineer")
            annual_income = st.number_input("Annual_Income", min_value=0.0, value=50000.0, step=1000.0)
            monthly_inhand_salary = st.number_input("Monthly_Inhand_Salary", min_value=0.0, value=4000.0, step=100.0)

        with col2:
            st.markdown("**Kredit & Pinjaman**")
            num_bank_accounts = st.number_input("Num_Bank_Accounts", min_value=0, max_value=20, value=3)
            num_credit_card = st.number_input("Num_Credit_Card", min_value=0, max_value=20, value=2)
            interest_rate = st.number_input("Interest_Rate", min_value=0.0, max_value=100.0, value=10.0)
            num_of_loan = st.number_input("Num_of_Loan", min_value=0, max_value=9, value=1)
            type_of_loan = st.text_input("Type_of_Loan", value="Personal Loan")
            credit_mix = st.selectbox("Credit_Mix", ["Bad", "Standard", "Good", "Unknown"], index=1)
            outstanding_debt = st.number_input("Outstanding_Debt", min_value=0.0, value=1500.0, step=100.0)

        with col3:
            st.markdown("**Perilaku Pembayaran**")
            delay_from_due_date = st.number_input("Delay_from_due_date", min_value=0, value=5)
            num_of_delayed_payment = st.number_input("Num_of_Delayed_Payment", min_value=0, value=2)
            changed_credit_limit = st.number_input("Changed_Credit_Limit", value=5.0)
            num_credit_inquiries = st.number_input("Num_Control_Inquiries", min_value=0, value=3)
            credit_utilization_ratio = st.number_input("Credit_Utilization_Ratio", min_value=0.0, max_value=100.0, value=30.0)
            credit_history_age = st.text_input("Credit_History_Age (format: 'X Years Y Months')", value="5 Years 3 Months")
            payment_of_min_amount = st.selectbox("Payment_of_Min_Amount", ["No", "NM", "Yes"], index=0)
            total_emi_per_month = st.number_input("Total_EMI_per_month", min_value=0.0, value=200.0)
            amount_invested_monthly = st.number_input("Amount_invested_monthly", min_value=0.0, value=300.0)
            payment_behaviour = st.text_input("Payment_Behaviour", value="Low_spent_Small_value_payments")
            monthly_balance = st.number_input("Monthly_Balance", value=400.0)

        submitted = st.form_submit_button("🔮 Prediksi")

    if submitted:
        record = {
            "Age": age,
            "Occupation": occupation,
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": monthly_inhand_salary,
            "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card,
            "Interest_Rate": interest_rate,
            "Num_of_Loan": num_of_loan,
            "Type_of_Loan": type_of_loan,
            "Delay_from_due_date": delay_from_due_date,
            "Num_of_Delayed_Payment": num_of_delayed_payment,
            "Changed_Credit_Limit": changed_credit_limit,
            "Num_Control_Inquiries": num_credit_inquiries,
            "Credit_Mix": credit_mix,
            "Outstanding_Debt": outstanding_debt,
            "Credit_Utilization_Ratio": credit_utilization_ratio,
            "Credit_History_Age": credit_history_age,
            "Payment_of_Min_Amount": payment_of_min_amount,
            "Total_EMI_per_month": total_emi_per_month,
            "Amount_invested_monthly": amount_invested_monthly,
            "Payment_Behaviour": payment_behaviour,
            "Monthly_Balance": monthly_balance,
        }

        try:
            with st.spinner("Memanggil endpoint..."):
                result = invoke_endpoint_json(record, region, endpoint_name)
            pred = result["predictions"][0]

            label = pred["predicted_label"]
            color = {"Good": "🟢", "Standard": "🟡", "Poor": "🔴"}.get(label, "⚪")
            st.success(f"{color} Prediksi Credit Score: **{label}**")

            if "probabilities" in pred:
                prob_df = pd.DataFrame.from_dict(pred["probabilities"], orient="index", columns=["Probability"])
                st.bar_chart(prob_df)

        except Exception as e:
            st.error(f"Gagal memanggil endpoint: {e}")

# ---------------------------------------------------------------------------
# TAB 2 — Batch prediction via CSV upload
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload CSV untuk Prediksi Batch")
    st.caption("CSV harus punya kolom mentah yang sama seperti data training (Age, Annual_Income, Occupation, dst).")

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"Preview data ({len(df_preview)} baris):")
        st.dataframe(df_preview.head(10))

        if st.button("🔮 Jalankan Prediksi Batch"):
            try:
                csv_buffer = io.StringIO()
                df_preview.to_csv(csv_buffer, index=False)

                with st.spinner("Memanggil endpoint untuk seluruh data..."):
                    result = invoke_endpoint_csv(csv_buffer.getvalue(), region, endpoint_name)

                preds = result["predictions"]
                result_df = df_preview.copy()
                result_df["predicted_label"] = [p["predicted_label"] for p in preds]
                for cls in ["Poor", "Standard", "Good"]:
                    if "probabilities" in preds[0]:
                        result_df[f"prob_{cls}"] = [p["probabilities"].get(cls) for p in preds]

                st.success(f"Selesai! {len(result_df)} baris berhasil diprediksi.")
                st.dataframe(result_df)

                st.download_button(
                    "⬇️ Download Hasil (CSV)",
                    data=result_df.to_csv(index=False).encode("utf-8"),
                    file_name="credit_score_predictions.csv",
                    mime="text/csv",
                )

                st.bar_chart(result_df["predicted_label"].value_counts())

            except Exception as e:
                st.error(f"Gagal memanggil endpoint: {e}")
