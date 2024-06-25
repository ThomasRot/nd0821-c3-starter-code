# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a LogisticRegression Model. 
Labels are binarized. Categorical Features are OneHotEncoded.

## Intended Use
Prediction for salary.

## Training Data
Publicly available Census Bureau data as provided in the exercise of udacity Course `Deploying a (sic!) ML model to Cloud Application Platform with FastApi`

## Evaluation Data
A subset of the original dataset with size roughly 20% of the full dataset.

## Metrics
precision=0.7107142857142857, recall=0.24875 and f1=0.3685185185185185

## Ethical Considerations
This model is processing personal data including race and gender. As it is sourced from the real world, real world biases may impact prediction on new data.

## Caveats and Recommendations
Re-Training on reduced selection of columns to avoid bias might be sensible.
Binarized Labels for salaries may not be useful.