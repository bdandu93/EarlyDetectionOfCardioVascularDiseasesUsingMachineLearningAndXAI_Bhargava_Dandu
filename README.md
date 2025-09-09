# 🫀 Heart Disease EDA & Prediction Project

This project focuses on combining, cleaning, and analyzing multiple Heart Disease datasets from Cleveland, Hungary, and Switzerland. 
The output is a cleaned dataset (heart_disease_combined.csv) and a series of exploratory data analysis (EDA) visualizations that help uncover key risk factors for heart disease.


## Project Structure

- [ ] data
- [ ] docs
- [ ] models
- [ ] notebooks
- [ ] outputs 
- [ ] requirements
- [ ] scripts

## Installing dependencies
```
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```
## Installing requirements
```
pip install -r requirements.txt

```
## Data Preparation and EDA
```
python3 scripts/EarlyCardioVascularDisease_EDA.py

python3 scripts/EarlyCardioVascularData_PreProcessing.py

```
This will 
- merge all the individual datasets into a single file.
- clean all the missing values and NaN values.
- Convert all the datatypes into
- It saves the final dataset into the combined_Dataset.csv


## Visualizations Included.

### Univariate analysis

- [ ] Gender distribution (donut chart)
- [ ] Age distribution (histogram)
- [ ] Chest pain types (count plot)
- [ ] Serum cholesterol (boxen plot)
- [ ] Fasting blood sugar (stem plot)
- [ ] ECG results (strip plot)
- [ ] Maximum heart rate (KDE plot)
- [ ] Exercise-induced angina (pie chart)
- [ ] Oldpeak ST depression (violin plot)
- [ ] Slope of peak exercise ST segment (bar chart)
- [ ] Number of major vessels (step plot)
- [ ] Thalassemia types (count plot)

### Bivariate analysis

- [ ] Correlation heatmap
- [ ] Pairplot with target labels
- [ ] Scatterplot: Age vs Max Heart Rate vs Target
- [ ] Histogram: Age distribution by gender


## Models Implementation

- [ ] Random Forest
- [ ] Ada Boost
- [ ] Gradient Boost


## Machine Learning Techniques used

- [ ] Hyperparameter Tuning
- [ ] Feature Extraction
- [ ] Feature Engineering
- [ ] Ensemble Learning

## XAI Techniques used

- [ ] SHAP
- [ ] LIME

## Results and evaluation

- [ ] Evaluating Model Performance.
- [ ] Evaluating outputs generated after model deployment.
- [ ] limitations of final model and further developments.

## Conclusion and Future work

- [ ] Conculsion
- [ ] Future work and discussions.

## Project Maintained By:

- [ ] Name: Bhargava Dandu.
- [ ] Student ID: D00274867.


## Project Supervised by:

- [ ] Dr. Zohiab Ijaz.
- [ ] Dr. Abhishek Kaushik.