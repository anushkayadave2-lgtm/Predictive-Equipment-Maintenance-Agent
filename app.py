import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier

st.set_page_config(
    page_title="Predictive Equipment Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("Predictive Equipment Maintenance")

st.markdown("""
Welcome to the Predictive Equipment Maintenance portal. Upload equipment sensor data in CSV format to get started.
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

REQUIRED_COLUMNS = ["timestamp", "temperature", "vibration", "pressure", "rpm", "power"]

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_columns:
            st.error(f"Invalid Dataset! Missing required column(s): **{', '.join(missing_columns)}**")
            st.info(f"Required columns: **{', '.join(REQUIRED_COLUMNS)}**")
        else:
            st.success(f"File uploaded successfully: **{uploaded_file.name}**")
            
            # --- Data Cleaning ---
            initial_rows = len(df)
            
            # 1. Remove duplicate rows
            duplicate_count = int(df.duplicated().sum())
            df = df.drop_duplicates()
            
            # 2. Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            
            # 3. Sort data by timestamp
            df = df.sort_values(by="timestamp").reset_index(drop=True)
            
            # 4. Handle missing sensor values
            sensor_cols = ["temperature", "vibration", "pressure", "rpm", "power"]
            missing_values_count = int(df[sensor_cols].isna().sum().sum())
            df[sensor_cols] = df[sensor_cols].ffill().bfill().fillna(df[sensor_cols].median())
            
            final_rows = len(df)
            
            # --- Data Quality Summary ---
            st.subheader("Data Quality Summary")
            q_col1, q_col2, q_col3, q_col4 = st.columns(4)
            with q_col1:
                st.metric("Total Rows", final_rows, delta=f"-{initial_rows - final_rows} removed" if initial_rows != final_rows else None)
            with q_col2:
                st.metric("Duplicates Removed", duplicate_count)
            with q_col3:
                st.metric("Missing Values Imputed", missing_values_count)
            with q_col4:
                st.metric("Total Columns", df.shape[1])
                
            if not df.empty:
                st.caption(f"Time Range: **{df['timestamp'].min()}** to **{df['timestamp'].max()}**")
                
            st.subheader("Cleaned Data Preview (First 5 Rows)")
            st.dataframe(df.head(5), use_container_width=True)
            
            # --- Abnormal Pattern Detection (IsolationForest) ---
            st.subheader("Abnormal Pattern Detection")
            feature_cols = ["temperature", "vibration", "pressure", "rpm", "power"]
            X_iso = df[feature_cols]
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            df["abnormal_flag"] = iso_forest.fit_predict(X_iso) == -1
            
            num_abnormal = int(df["abnormal_flag"].sum())
            total_readings = len(df)
            pct_abnormal = (num_abnormal / total_readings * 100) if total_readings > 0 else 0.0
            
            col_ab1, col_ab2 = st.columns(2)
            with col_ab1:
                st.metric("Abnormal Readings", num_abnormal)
            with col_ab2:
                st.metric("Percentage Abnormal", f"{pct_abnormal:.1f}%")
                
            if num_abnormal > 0:
                st.warning(f"⚠️ **Warning**: {num_abnormal} abnormal reading(s) ({pct_abnormal:.1f}%) detected in equipment sensor data!")
            else:
                st.success("✅ No abnormal patterns detected. Sensor telemetry is operating normally.")
                
            # --- RandomForest Failure Prediction & Risk Assessment ---
            st.subheader("RandomForest Failure Risk Assessment")
            if "failure" in df.columns:
                X_rf = df[feature_cols]
                y_rf = df["failure"]
                
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_rf, y_rf)
                
                df["predicted_failure"] = rf_model.predict(X_rf)
                
                # Model failure probability as percentage
                if 1 in rf_model.classes_:
                    fail_idx = list(rf_model.classes_).index(1)
                    df["failure_prob"] = rf_model.predict_proba(X_rf)[:, fail_idx] * 100.0
                else:
                    df["failure_prob"] = 0.0
                    
                def get_risk_level(prob):
                    if prob <= 30.0:
                        return "Low"
                    elif prob <= 60.0:
                        return "Medium"
                    elif prob <= 80.0:
                        return "High"
                    else:
                        return "Critical"
                        
                df["risk_level"] = df["failure_prob"].apply(get_risk_level)
                
                avg_prob = float(df["failure_prob"].mean())
                max_prob = float(df["failure_prob"].max())
                overall_risk = get_risk_level(max_prob)
                
                total_pred_failures = int(df["predicted_failure"].sum())
                
                rf_col1, rf_col2, rf_col3, rf_col4 = st.columns(4)
                with rf_col1:
                    st.metric("Predicted Failures", total_pred_failures)
                with rf_col2:
                    st.metric("Avg Failure Prob.", f"{avg_prob:.1f}%")
                with rf_col3:
                    st.metric("Max Failure Prob.", f"{max_prob:.1f}%")
                with rf_col4:
                    st.metric("Peak Risk Level", overall_risk)
                    
                def get_maintenance_alert(risk):
                    if risk == "Critical":
                        return "Urgent maintenance inspection recommended."
                    elif risk == "High":
                        return "Maintenance recommended."
                    elif risk == "Medium":
                        return "Monitor machine condition."
                    else:
                        return "Machine operating normally."
                        
                df["maintenance_alert"] = df["risk_level"].apply(get_maintenance_alert)
                
                if overall_risk == "Critical":
                    st.error("🔴 **Maintenance Alert**: Urgent maintenance inspection recommended.")
                elif overall_risk == "High":
                    st.error("🟠 **Maintenance Alert**: Maintenance recommended.")
                elif overall_risk == "Medium":
                    st.warning("🟡 **Maintenance Alert**: Monitor machine condition.")
                else:
                    st.success("🟢 **Maintenance Alert**: Machine operating normally.")
                    
                st.markdown("##### Failure Probability & Risk Breakdown Preview")
                display_df = df.copy()
                display_df["failure_prob_pct"] = display_df["failure_prob"].map(lambda x: f"{x:.1f}%")
                preview_cols = ["timestamp", "temperature", "vibration", "pressure", "rpm", "power", "failure", "predicted_failure", "failure_prob_pct", "risk_level", "maintenance_alert"]
                st.dataframe(display_df[[col for col in preview_cols if col in display_df.columns]].head(10), use_container_width=True)
            else:
                st.info("ℹ️ 'failure' column not present in CSV. Add a 'failure' column to enable supervised failure prediction.")


            
            # --- Single Sensor Selection & Statistics ---
            st.subheader("Interactive Sensor Analysis")
            selected_sensor = st.selectbox(
                "Select Sensor to View:",
                options=["temperature", "vibration", "pressure", "rpm", "power"],
                format_func=lambda x: x.capitalize()
            )
            
            # Statistics calculation
            s_series = df[selected_sensor]
            s_mean = s_series.mean()
            s_min = s_series.min()
            s_max = s_series.max()
            s_std = s_series.std()
            
            # Trend calculation using Scikit-Learn LinearRegression
            if len(df) > 1:
                X_mat = np.arange(len(df)).reshape(-1, 1)
                y_vec = s_series.values
                model = LinearRegression()
                model.fit(X_mat, y_vec)
                total_change = model.coef_[0] * len(df)
                std_denom = s_std if (s_std is not None and s_std > 0) else 1.0
                relative_change = total_change / std_denom
                
                if relative_change > 0.15:
                    trend_status = "Increasing 📈"
                elif relative_change < -0.15:
                    trend_status = "Decreasing 📉"
                else:
                    trend_status = "Stable ➡️"
            else:
                trend_status = "Stable ➡️"
                
            st1, st2, st3, st4, st5 = st.columns(5)
            with st1:
                st.metric("Mean", f"{s_mean:.2f}")
            with st2:
                st.metric("Min", f"{s_min:.2f}")
            with st3:
                st.metric("Max", f"{s_max:.2f}")
            with st4:
                st.metric("Std Dev", f"{s_std:.2f}")
            with st5:
                st.metric("Overall Trend", trend_status)
                
            fig_selected = px.line(
                df,
                x="timestamp",
                y=selected_sensor,
                title=f"{selected_sensor.capitalize()} Over Time",
                labels={"timestamp": "Timestamp", selected_sensor: selected_sensor.capitalize()}
            )
            st.plotly_chart(fig_selected, use_container_width=True, key="selected_sensor")
            
            # --- Sensor Line Charts (Tabs) ---
            st.subheader("All Sensor Telemetry Trends")
            tabs = st.tabs(["Temperature", "Vibration", "Pressure", "RPM", "Power"])
            sensor_metrics = ["temperature", "vibration", "pressure", "rpm", "power"]
            
            for tab, metric in zip(tabs, sensor_metrics):
                with tab:
                    fig = px.line(
                        df,
                        x="timestamp",
                        y=metric,
                        title=f"{metric.capitalize()} Over Time",
                        labels={"timestamp": "Timestamp", metric: metric.capitalize()}
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"sensor_trends_{metric}")
            
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")










