This repository contains scripts for crop yield prediction and irrigation water estimation using both:

a Machine Learning (ML) approach, and
a process-based model (AquaCrop-OSPy).

The ML workflow focuses on data-driven prediction, while AquaCrop-OSPy is used for simulation-based analysis and irrigation optimization. Together, they allow comparison between empirical and process-based modeling approaches.

⚙️ Workflow Summary

The repository follows a typical pipeline:

Data preprocessing and exploration
Feature engineering and dataset preparation
Machine learning model development (yield prediction)
Process-based modeling (AquaCrop-OSPy)
Optimization and benchmarking
Visualization and result analysis

📂 File Descriptions

🔹 Data_Analysis_(Preprocessing).ipynb
Performs data cleaning and preprocessing
Explores data distributions
Handles missing values and inconsistencies
Separates datasets for:
yield prediction (ML models)
irrigation and environmental inputs
Prepares final datasets for modeling

🔹 Yield_Prediction.ipynb
Implements machine learning models for crop yield prediction
Uses processed datasets from preprocessing step
Includes model training, validation, and evaluation
🔹 Irrigation_Modeling.ipynb
Uses AquaCrop-OSPy for simulating crop growth and irrigation
Estimates crop water use and irrigation requirements
Supports scenario-based analysis

🔹 Aq_Optimization_functons.py
Contains functions for optimization of irrigation parameters
Supports multi-objective optimization (e.g., yield vs water use)
Designed to work with AquaCrop simulation outputs
🔹 Benchmark_and_Comparison.ipynb
Compares ML predictions vs AquaCrop simulations
Evaluates performance across different scenarios
Helps assess strengths and limitations of each approach

🔹 Pareto_plotting.ipynb
Generates Pareto fronts for multi-objective optimization
Visualizes trade-offs (e.g., yield vs irrigation water use)

🔹 plot_functions.py
Utility functions for custom plotting and visualization
Used across notebooks for consistent figure generation

🔹 Fields excluded due to ET mismatch.ipynb
Identifies and analyzes fields excluded due to evapotranspiration inconsistencies
Ensures data quality for modeling
