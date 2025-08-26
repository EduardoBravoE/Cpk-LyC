# utils/style.py
import streamlit as st

def apply_theme():
    st.markdown(
        """
        <style>
        /* Cards de métricas */
        .metric-card {
            background: #ffffff;
            border: 1px solid #E6E6E6;
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .small-muted { color: #6b7280; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True
    )
