#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Configure path for Python to find 'src'
export PYTHONPATH=$PWD

# Run Streamlit app
streamlit run src/frontend/app.py
