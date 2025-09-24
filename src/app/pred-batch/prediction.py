import pickle

from etl import UserGenerator
from feature_engineer import FeatureEngineer


def get_model():
    with open('models/model_rf.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def get_etl_data():
    user_generator = UserGenerator(n_samples=100000)
    df = user_generator.create_dataset()
    return df

def get_data():
    df = get_etl_data()
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.create_features()
    return df

def predict(model, df):
    return model.predict_proba(df)

def save_prediction(df, prediction):
    df["prediction"] = prediction[:, 1]
    df["prediction"] = df["prediction"].astype(int)
    df["prediction"] = df["prediction"].map({0: "No", 1: "Si"})
    df.to_csv('predictions/predictions.csv', index=False)
    return df

def save_prediction_proba(df, prediction):
    df["prediction_0"] = prediction[:, 0]
    df["prediction_1"] = prediction[:, 1]
    df["prediction_0"] = df["prediction_0"].astype(int)
    df["prediction_1"] = df["prediction_1"].astype(int)
    df["prediction_0"] = df["prediction_0"].map({0: "No", 1: "Si"})
    df["prediction_1"] = df["prediction_1"].map({0: "No", 1: "Si"})
    df.to_csv('predictions/predictions_proba.csv', index=False)
    return df

def main():
    model = get_model()
    df = get_data()
    prediction = predict(model, df)
    #save_prediction(df, prediction)
    save_prediction_proba(df, prediction)

if __name__ == "__main__":
    main()