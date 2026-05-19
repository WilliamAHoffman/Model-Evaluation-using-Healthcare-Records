import pandas as pd

def load_data():
    diabetes_csv = pd.read_csv(
        'diabetes_012_health_indicators_BRFSS2015.csv'
    )

    # Convert decimal values to integers
    diabetes_csv = diabetes_csv.astype(int)

    # Combine pre-diabetes and diabetes into one class
    diabetes_csv['Diabetes_012'] = (
        diabetes_csv['Diabetes_012'].replace(2, 1)
    )

    y = diabetes_csv['Diabetes_012']
    X = diabetes_csv.drop(columns='Diabetes_012')

    return y, X

