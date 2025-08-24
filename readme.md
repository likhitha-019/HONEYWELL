# F&B Process Anomaly Prediction System

## Project Overview
This project develops a multi-variable predictive maintenance system for an industrial bakery process. It uses machine learning to predict final product quality anomalies based on real-time process parameter data, enabling proactive intervention.

## Hackathon Deliverables Addressed
1.  **F&B Process Identification:** Detailed breakdown of bakery process steps, equipment, and control parameters.
2.  **References:** Citations from academic and industry sources.
3.  **Data Streams:** Identification and justification of relevant time-series and discrete data streams.
4.  **Quality Definition:** Quantitative definition of final product quality.
5.  **Multivariable Prediction Model:** A Random Forest model built in Python.
6.  **Real-time Dashboard:** A Streamlit dashboard for process visualization and quality prediction.

## How to Run
1.  Ensure Python 3.8+ is installed.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the application: `streamlit run main_app.py`
4.  The app will open in your browser at `http://localhost:8501`.

## Project Structure
- `main_app.py`: The main application containing data generation, ML model, and dashboard.
- `process_overview.py`: Detailed documentation of the bakery process and theory.
- `requirements.txt`: Python package dependencies.

## Model Performance
The Random Forest model achieves an R² score of ~0.92, effectively capturing the relationship between process parameters and final product quality. Key drivers of quality are baking temperature, fermentation time, and mixing time.