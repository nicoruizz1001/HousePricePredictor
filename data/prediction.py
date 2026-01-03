import os
import joblib
import pandas as pd


def ask_float_in_range(name, min_val, max_val):
    while True:
        try:
            value = float(input(f"{name} [{min_val}, {max_val}]: ").strip())
            if min_val <= value <= max_val:
                return value
            print(f"Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid number.")


def ask_positive_float(name):
    while True:
        try:
            value = float(input(f"{name}: ").strip())
            if value > 0:
                return value
            print("Please enter a value greater than 0.")
        except ValueError:
            print("Please enter a valid number.")


def ask_optional_float(name, hint=None):
    prompt = f"{name}"
    if hint:
        prompt += f" ({hint})"
    prompt += " [press Enter to skip]: "

    while True:
        s = input(prompt).strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            print("Please enter a valid number, or press Enter to skip.")


def ask_choice(name, choices):
    choices_str = ", ".join(choices)
    while True:
        val = input(f"{name} ({choices_str}): ").strip()
        if val in choices:
            return val
        print(f"Please choose one of: {choices_str}")


def ask_user_for_house_features(artifacts):
    raw = {}

    raw["longitude"] = ask_float_in_range("Longitude", -124.35, -114.31)
    raw["latitude"] = ask_float_in_range("Latitude", 32.54, 41.95)

    raw["total_rooms"] = ask_positive_float("About how many total rooms?")
    raw["total_bedrooms"] = ask_positive_float("How many bedrooms?")

    raw["housing_median_age"] = None

    raw["households"] = ask_optional_float("Households", "usually unknown")
    raw["population"] = ask_optional_float("Population", "usually unknown")
    raw["median_income"] = ask_optional_float("Median income", "usually unknown")

    train_categories = artifacts.get("train_categories")
    if train_categories is None:
        train_categories = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

    raw["ocean_proximity"] = ask_choice("Ocean proximity", train_categories)

    return raw


def preprocess_single_input(raw_input, artifacts, model=None):
    num_cols = artifacts.get("num_cols")

    if num_cols is None:
        for k in ["num_medians", "num_means", "num_stds"]:
            if k in artifacts and artifacts[k] is not None:
                obj = artifacts[k]
                if hasattr(obj, "keys"):
                    num_cols = list(obj.keys())
                else:
                    num_cols = list(obj.index)
                break

    if num_cols is None:
        num_cols = [
            "longitude", "latitude", "housing_median_age",
            "total_rooms", "total_bedrooms",
            "population", "households", "median_income"
        ]

    X_num = pd.DataFrame([{col: raw_input.get(col, None) for col in num_cols}])
    X_num = X_num.apply(pd.to_numeric, errors="coerce")

    num_medians = artifacts.get("num_medians")
    if num_medians is not None:
        X_num = X_num.fillna(pd.Series(num_medians))

    num_means = artifacts.get("num_means")
    num_stds = artifacts.get("num_stds")
    if num_means is not None and num_stds is not None:
        X_num_scaled = (X_num - pd.Series(num_means)) / pd.Series(num_stds).replace(0, 1)
    else:
        X_num_scaled = X_num.copy()

    train_categories = artifacts.get("train_categories")
    if train_categories is None:
        train_categories = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

    X_cat = pd.DataFrame(0, index=[0], columns=train_categories)
    chosen = raw_input.get("ocean_proximity")
    if chosen in X_cat.columns:
        X_cat.loc[0, chosen] = 1

    cat_columns = artifacts.get("cat_columns")
    if cat_columns is not None:
        if all(col.startswith("ocean_proximity_") for col in cat_columns):
            X_cat.columns = [f"ocean_proximity_{c}" for c in X_cat.columns]
        X_cat = X_cat.reindex(columns=cat_columns, fill_value=0)

    X_one = pd.concat([X_num_scaled, X_cat], axis=1)

    if model is not None and hasattr(model, "feature_names_in_"):
        X_one = X_one.reindex(columns=list(model.feature_names_in_), fill_value=0)

    return X_one


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(here, "..", "models"))

    artifacts = joblib.load(os.path.join(models_dir, "preprocess_artifacts.joblib"))
    model = joblib.load(os.path.join(models_dir, "random_forest_model.joblib"))

    raw_input = ask_user_for_house_features(artifacts)
    X_one = preprocess_single_input(raw_input, artifacts, model=model)

    prediction = model.predict(X_one)[0]
    print(f"Predicted median house value: ${prediction:,.0f}")


if __name__ == "__main__":
    main()
