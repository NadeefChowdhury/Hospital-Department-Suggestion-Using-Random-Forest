# Hospital-Department-Suggestion-Using-Random-Forest
An app that takes input from the user about symptoms of diseases and suggests the user which department of the hospital they should visit.

## Data
The Bangladesh Triage Dataset is a synthetic dataset consisting of 300 rows, 12 columns containing severity (0 to 1) of symptoms, and the department of hospital the patient should visit

## AI Model
A Random Forest Classifier was from Scikit-learn was trained on the data and used to suggest the user the department. However, since medical suggestions are a sensitive issue, if the confidence of the model is below 40%, the app suggests the user to visit General Medicine department.

Link of the app: [https://hospital-department-suggestion-using-random-forest.streamlit.app/](https://hospital-department-suggestion-using-random-forest.streamlit.app/)
