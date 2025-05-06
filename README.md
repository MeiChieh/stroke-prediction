# 🧠 Stroke Prediction Analysis

## 📊 Overview

This project analyzes a stroke prediction dataset to build models that can identify patients at risk of stroke based on various health parameters. The models can assist healthcare professionals in diagnosing stroke risk and providing preventive recommendations to high-risk individuals.

The most performant model is encapsulated in an app, it can be accessed through the FastAPI endpoint after running the Docker container.

## 📚 Dataset

The analysis uses the Stroke Prediction dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), which includes:

- Patient demographics (gender, age)
- Medical history (various diseases)
- Lifestyle factors (smoking status)
- Other health-related parameters

## 📗 Notebooks

- [1_cleaning_and_eda.ipynb](https://github.com/MeiChieh/stroke-prediction/blob/main/1_cleaning_and_eda.ipynb): Contains the exploratory data analysis and initial insights
- [2_project_modeling.ipynb](https://github.com/MeiChieh/stroke-prediction/blob/main/2_modeling.ipynb): Includes model development, evaluation, and selection

## 📈 Analysis Structure

### 1. Data Analysis & Preprocessing

- Null value detection and handling
- Duplicate record identification
- Feature distribution analysis across target classes
- Correlation analysis between features

### 2. Feature Engineering

- Data imputation based on EDA insights
- Creation of new features using:
  - Domain knowledge
  - Exploratory data analysis
  - Statistical heuristics

### 3. Model Development

- Comprehensive model evaluation using appropriate scoring metrics
- Hyperparameter tuning and optimization
- Cross-validation implementation
- Learning curve analysis
- Error analysis and model interpretation
- Model testing and validation

### 4. Deployment

- Containerization of the best performing model using Docker
- API development using FastAPI
- Production deployment

## ⭐ Key Findings
- `Age` plays the most crucial role for stroke detection in this dataset,
as the instances with stroke almost only occurs in elder people, it is hard for all models to identify stroke precisely in the elderly group, thus almost 1/3 of data is falsely classified as having stroke.

- If we want to better distinguish the elderly with stroke, we need to have more features.

- Most models can identify around 85% of stroke patients. 

- Logit and LGBM have the best performance, Logit model requires a smaller sample size to perform well (logit: 1000~, LGBM: 1700~), while LGBM has slightly lower FP rate (3%).

## 📁 Project Structure

```
├── project.ipynb # EDA and initial analysis
├── project_modeling.ipynb # Model development and evaluation
├── stroke_prediction_app/ # Deployment application
├── requirements.txt # Python package requirements
└── README.md # Project documentation
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Docker

## 🚀 Setup and Installation

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. For model deployment:
   ```bash
   cd stroke_prediction_app
   docker build -t stroke-prediction .
   docker run -p 8000:8000 stroke-prediction
   ```

## 🔄 Future Improvements

1. Implement additional feature engineering techniques
2. Explore more advanced machine learning models
3. Add real-time monitoring for model performance
4. Develop a user-friendly web interface
5. Implement model versioning and A/B testing capabilities

## 📦 Dependencies

Key dependencies include:

- numpy
- pandas
- scikit-learn
- fastapi
- docker

For a complete list of dependencies, see `requirements.txt`.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Mei-Chieh Chien
