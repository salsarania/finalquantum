import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
@st.cache
def load_model():
    with open('model_regresi.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Main function
def main():
    st.title('Delivery Time Prediction')
    st.subheader('To predict the preparation time of an order from the moment it is received until it is ready for delivery')

    # Sidebar inputs
    st.sidebar.header('Input Features')

    # Load input features
    total_items = st.sidebar.number_input('Total Items')
    min_item_price = st.sidebar.number_input('Minimum Item Price')
    max_item_price = st.sidebar.number_input('Maximum Item Price')
    subtotal = st.sidebar.number_input('Subtotal')
    num_distinct_items = st.sidebar.number_input('Number of Distinct Items')
    total_outstanding_orders = st.sidebar.number_input('Total Outstanding Orders')
    total_onshift_partners = st.sidebar.number_input('Total Onshift Partners')
    total_busy_partners = st.sidebar.number_input('Total Busy Partners')
    market_id = st.sidebar.number_input('Market ID', min_value=1, max_value=6, step=1)
    order_protocol = st.sidebar.number_input('Order Protocol', min_value=1, max_value=7, step=1)

    # Combine input features into array
    input_features = np.array([[total_items, min_item_price, max_item_price, subtotal, num_distinct_items, total_outstanding_orders, total_onshift_partners, total_busy_partners, market_id, order_protocol]])

    # Load model
    model = load_model()

    # Make prediction
    prediction = model.predict(input_features)

    # Transform back to minutes after log transformation
    delivery_time_minutes = np.exp(prediction)

    # Display prediction
    st.subheader('Predicted Delivery Time (in minutes)')
    st.write(delivery_time_minutes)

if __name__ == '__main__':
    main()
