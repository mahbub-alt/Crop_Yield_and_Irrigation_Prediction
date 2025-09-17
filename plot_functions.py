import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error,make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
import shap
from scipy.stats import pearsonr
from sklearn.model_selection import KFold


def pbias(predicted, observed):
    return 100.0 * np.sum(predicted - observed ) / np.sum(observed)

def nmae(predicted, observed):
    return 100 * mean_absolute_error(observed, predicted) / np.mean(observed)

def evaluate_model(
    y_train=None, y_pred_train=None,
    y_test=None, y_pred_test=None,
    y_holdout=None, y_pred_holdout=None,
    xlabel='Observed', ylabel='Predicted',
    units='', xlim=(3.5, 18), ylim=(3.5, 18),
    hue=None, text="Train", label = None,
    p =.35, # x axis location of validation text , could be .02, .35 or .65
    legend_outside=False
):

    metrics = {}

    plt.figure(figsize=(7.5, 5.5))
    

    # --- Train Plot ---
    if y_train is not None and y_pred_train is not None:
        metrics["Train"] = {
            "rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "nmae": nmae(y_pred_train, y_train),
            "pbias": pbias(y_pred_train, y_train),
            "r2": r2_score(y_train, y_pred_train)
        }

        sns.scatterplot(x=y_train, y=y_pred_train, label="Train", color="blue", edgecolor="k")
        sns.regplot(x=y_train, y=y_pred_train, scatter=False, ci=None, color="blue",label="Train regression line" )

        plt.annotate(
            f"Train Set:\nRMSE: {metrics['Train']['rmse']:.2f}\n"
            f"PBIAS: {metrics['Train']['pbias']:.2f}%\n"
            f"NMAE: {metrics['Train']['nmae']:.2f}%\n"
            f"$R^2$: {metrics['Train']['r2']:.2f}",
            xy=(0.02, 0.80), xycoords='axes fraction', fontsize=11
        )

    # --- Test Plot ---
    if y_test is not None and y_pred_test is not None:
        metrics["Test"] = {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "nmae": nmae(y_pred_test, y_test),
            "pbias": pbias(y_pred_test, y_test),
            "r2": r2_score(y_test, y_pred_test)
        }

        sns.scatterplot(x=y_test, y=y_pred_test, label="Test", color="red", edgecolor="k")
        sns.regplot(x=y_test, y=y_pred_test, scatter=False, ci=None, color="red", label="Test regression line")

        plt.annotate(
            f"Test Set:\nRMSE: {metrics['Test']['rmse']:.2f}\n"
            f"PBIAS: {metrics['Test']['pbias']:.2f}%\n"
            f"NMAE: {metrics['Test']['nmae']:.2f}%\n"
            f"$R^2$: {metrics['Test']['r2']:.2f}",
            xy=(0.65, 0.80), xycoords='axes fraction', fontsize=11
        )

    # --- Holdout Plot ---
    if y_holdout is not None and y_pred_holdout is not None:
        metrics["Holdout"] = {
            "rmse": np.sqrt(mean_squared_error(y_holdout, y_pred_holdout)),
            "nmae": nmae(y_pred_holdout, y_holdout),
            "pbias": pbias(y_pred_holdout, y_holdout),
            "r2": r2_score(y_holdout, y_pred_holdout)
        }
        


        sns.scatterplot(x=y_holdout, y=y_pred_holdout, label=label, color="red", edgecolor="k", hue=hue)
        sns.regplot(x=y_holdout, y=y_pred_holdout, scatter=False, ci=None, color="red", label="Holdout regression line")

        plt.annotate(
            f"{text}:\nRMSE: {metrics['Holdout']['rmse']:.2f}\n"
              f"PBIAS: {metrics['Holdout']['pbias']:.2f}%\n"
            f"NMAE: {metrics['Holdout']['nmae']:.2f}%\n"
          
            f"$R^2$: {metrics['Holdout']['r2']:.2f}",
            xy=(p, 0.80), xycoords='axes fraction', fontsize=11
        )

    # Axis and layout
    plt.xlabel(f"{xlabel} ({units})", fontsize=12)
    plt.ylabel(f"{ylabel} ({units})", fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axline((0, 0), slope=1, label='1:1 line', color='gray', linestyle='--')
    
    if legend_outside:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    else:
        plt.legend(loc='lower right')
        
    plt.tight_layout()
    plt.show()

#     return metrics


# --- kFold----


def run_xgb_kfold(
    X, 
    y, 
    n_splits=5, 
    random_state=153, 
    params=None, 
    df=None, 
    field_col="FieldID", 
    year_col="Year"
):
    """
    Run XGBoost regression with KFold CV, tracking FieldID and Year.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_splits : int, default=5
        Number of folds in KFold CV.
    random_state : int, default=153
        Random seed for reproducibility.
    params : dict, optional
        XGBoost parameters. If None, uses default small set.
    df : pd.DataFrame, optional
        Original dataframe containing FieldID and Year.
    field_col : str, default="FieldID"
        Column name for field IDs.
    year_col : str, default="Year"
        Column name for years.

    Returns
    -------
    results_df : pd.DataFrame
        Predictions and actuals with columns:
        ['Fold', 'Set', 'Actual', 'Predicted', 'FieldID', 'Year']
    metrics_df : pd.DataFrame
        MAE and R² per fold.
    """
    if params is None:
        params = {
            'n_estimators': 50,
            'max_depth': 2,
            'learning_rate': 0.2,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'gamma': 0.6,
            'reg_alpha': 0.4,
            'random_state': random_state
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    results = []
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Fit model
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Grab metadata (if df provided, year and fieldid)
        if df is not None:
            train_fields = df.iloc[train_idx][field_col].values
            train_years = df.iloc[train_idx][year_col].values
            test_fields = df.iloc[test_idx][field_col].values
            test_years = df.iloc[test_idx][year_col].values
        else:
            train_fields = train_years = [None] * len(train_idx)
            test_fields = test_years = [None] * len(test_idx)

        # Save results
        results.extend([
            pd.DataFrame({
                'Fold': fold,
                'Set': 'Train',
                'Actual': y_train.values,
                'Predicted': y_train_pred,
                'FieldID': train_fields,
                'Year': train_years
            }),
            pd.DataFrame({
                'Fold': fold,
                'Set': 'Test',
                'Actual': y_test.values,
                'Predicted': y_test_pred,
                'FieldID': test_fields,
                'Year': test_years
            })
        ])

        # Save metrics
        metrics.append({
            'Fold': fold,
            'Train_MAE': mean_absolute_error(y_train, y_train_pred),
            'Test_MAE': mean_absolute_error(y_test, y_test_pred),
            'Train_R2': r2_score(y_train, y_train_pred),
            'Test_R2': r2_score(y_test, y_test_pred)
        })

    results_df = pd.concat(results, ignore_index=True)
    metrics_df = pd.DataFrame(metrics)
    return results_df, metrics_df


# ------


def calculate_yearly_metrics(
    df,
    group_by_field=False,
    calculate_yield=True,
    calculate_irrigation=True,
    model=None,
    estimated="Simulated"
):
    """
    Calculate RMSE, PBIAS, and NMAE for yield and/or irrigation,
    grouped by Year or FieldID, and optionally tagged with a model name.

    Parameters:
    - df: DataFrame with required columns like:
        ['Year', 'FieldID', 'Reported_Yield', 'Simulated_Yield', ...]
    - group_by_field: Group by 'FieldID' instead of 'Year'
    - calculate_yield: Whether to calculate metrics for yield
    - calculate_irrigation: Whether to calculate metrics for irrigation
    - model: Optional string tag to label the model
    - estimated: Column prefix for simulated or predicted values (default: 'Simulated')

    Returns:
    - A tidy DataFrame with columns like:
        ['Year' or 'FieldID', 'Metric', 'Yield', 'Irrigation', 'Model' (if provided)]
    """

    group_cols = ['FieldID'] if group_by_field else ['Year']
    grouped = df.groupby(group_cols)
    results = []

    # Column names
    sim_yield_col = f"{estimated}_Yield"
    sim_irr_col = f"{estimated}_Irrigation"

    for keys, group in grouped:
        # Fix single-element tuple key from groupby
        if isinstance(keys, tuple) and len(keys) == 1:
            keys = keys[0]

        base = {"FieldID": keys} if group_by_field else {"Year": keys}
        if model:
            base["Model"] = model
        record = {}

        # --- Yield metrics ---
        if calculate_yield:
            yield_data = group[['Reported_Yield', sim_yield_col]].dropna()
            if not yield_data.empty:
                obs_y = yield_data['Reported_Yield'].values
                sim_y = yield_data[sim_yield_col].values
#                 print(len(obs_y))
                rmse_y = np.sqrt(mean_squared_error(obs_y, sim_y))
                pbias_y = 100 * np.sum(sim_y - obs_y) / np.sum(obs_y)
                nmae_y = 100 * mean_absolute_error(obs_y, sim_y) / np.mean(obs_y)
            else:
                rmse_y = pbias_y = nmae_y = np.nan

            record["RMSE"] = {"Yield": rmse_y}
            record["PBIAS"] = {"Yield": pbias_y}
            record["NMAE"] = {"Yield": nmae_y}

        # --- Irrigation metrics ---
        if calculate_irrigation:
            irr_data = group[['Reported_Irrigation', sim_irr_col]].dropna()
            if not irr_data.empty:
                obs_i = irr_data['Reported_Irrigation'].values
                sim_i = irr_data[sim_irr_col].values

                rmse_i = np.sqrt(mean_squared_error(obs_i, sim_i))
                pbias_i = 100 * np.sum( sim_i - obs_i) / np.sum(obs_i)
                nmae_i = 100 * mean_absolute_error(obs_i, sim_i) / np.mean(obs_i)
            else:
                rmse_i = pbias_i = nmae_i = np.nan

            if "RMSE" in record: record["RMSE"]["Irrigation"] = rmse_i
            else: record["RMSE"] = {"Irrigation": rmse_i}

            if "PBIAS" in record: record["PBIAS"]["Irrigation"] = pbias_i
            else: record["PBIAS"] = {"Irrigation": pbias_i}

            if "NMAE" in record: record["NMAE"]["Irrigation"] = nmae_i
            else: record["NMAE"] = {"Irrigation": nmae_i}

        # --- Format result ---
        for metric, values in record.items():
            row = base.copy()
            row["Metric"] = metric
            if calculate_yield:
                row["Yield"] = values.get("Yield", np.nan)
            if calculate_irrigation:
                row["Irrigation"] = values.get("Irrigation", np.nan)
            results.append(row)

    return pd.DataFrame(results)


# ----  Holdout model function ----

def evaluate_xgboost_holdout(
    df,
    feature_start_col: int = 3,
    target_col: str = "Reported_Yield",
    group_col: str = "Year",  
    param_grid: dict = None,
    random_state: int = 151,
    test_size: float = 0.2,
    use_test_split: bool = False,
    filter_features: bool = False,
    selected_features: list = None,
    skip_single_fields: bool = True
):
    """
    Evaluate XGBoost using a group-based holdout approach.

    For each unique value in `group_col`, that group is used as the holdout set,
    while the remaining data is used to train the model. Optionally performs 
    a train/test split within the training set. Results are collected for 
    training, test (if used), and holdout predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features, target, and grouping column.

    feature_start_col : int, default=3
        Column index from which feature columns start (if not using selected_features).

    target_col : str, default="Reported_Yield"
        Name of the target column.

    group_col : str, default="Year"
        Column name used for grouping the holdout (e.g., "Year" or "FieldID").

    param_grid : dict, optional
        Parameter grid for GridSearchCV. If None, a default small grid is used.

    random_state : int, default=151
        Random seed for reproducibility.

    test_size : float, default=0.2
        Proportion of data to use for test split when use_test_split=True.

    use_test_split : bool, default=False
        If True, split the training set into train/test. Otherwise, use all non-holdout data for training.

    filter_features : bool, default=False
        If True, use only selected_features for training. Otherwise, use all columns after feature_start_col.

    selected_features : list, optional
        List of feature names to use if filter_features=True.

    skip_single_fields : bool, default=True
        If True, skip groups (by group_col) that have 1 or fewer samples.
        If False, include them in evaluation, but note results may be unstable.

    Returns
    -------
    train_df : pd.DataFrame
        Predictions vs actuals for training data.

    test_df : pd.DataFrame or None
        Predictions vs actuals for test data (if use_test_split=True).

    holdout_df : pd.DataFrame
        Predictions vs actuals for each holdout group.
    """

    # --- Feature & target selection ---
    if filter_features and selected_features is not None:
        X_full = df[selected_features]
    else:
        X_full = df.iloc[:, feature_start_col:]

    y_full = df[target_col]
    groups = df[group_col]

    # --- Default parameter grid if not provided ---
    if param_grid is None:
        param_grid = {
            'n_estimators': [50],
            'max_depth': [2],
            'learning_rate': [0.2],
            'subsample': [0.9],
            'colsample_bytree': [0.7],
            'gamma': [0.6],
            'reg_alpha': [0.4]
        }

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    xgb_regressor = xgb.XGBRegressor()

    holdout_results, train_results, test_results = [], [], []

    # --- Loop over unique groups ---
    unique_values = df[group_col].unique()

    for val in unique_values:
        mask = (groups == val)

        # Optionally skip very small sets
        if skip_single_fields and mask.sum() <= 1:
            print(f"Skipping {group_col}={val} (only {mask.sum()} samples).")
            continue

        # Define holdout and training sets
        X_holdout = X_full[mask]
        y_holdout = y_full[mask]
        X_rest = X_full[~mask]
        y_rest = y_full[~mask]

        print(f"\n=== Holdout {group_col}: {val} ({mask.sum()} samples) ===")

        if use_test_split:
            X_train, X_test, y_train, y_test = train_test_split(
                X_rest, y_rest, test_size=test_size, random_state=random_state
            )
        else:
            X_train, y_train = X_rest, y_rest
            X_test, y_test = None, None

        # --- Hyperparameter search ---
        grid_search = GridSearchCV(
            estimator=xgb_regressor,
            param_grid=param_grid,
            scoring=mae_scorer,
            cv=5,
            n_jobs=8
        )
        grid_search.fit(X_train, y_train)

        best_model = xgb.XGBRegressor(**grid_search.best_params_)
        best_model.fit(X_train, y_train)

        # --- Predictions ---
        y_pred_train = best_model.predict(X_train)
        y_pred_holdout = best_model.predict(X_holdout)

        holdout_mae = mean_absolute_error(y_holdout, y_pred_holdout)
        holdout_pbias_val = pbias(y_pred_holdout, y_holdout)
        holdout_r2 = r2_score(y_holdout, y_pred_holdout)

        print(f"Holdout MAE: {holdout_mae:.4f}")
        print(f"Holdout PBIAS: {holdout_pbias_val:.2f}%")
        print(f"Holdout R2: {holdout_r2:.2f}")

        # --- Store results ---
        holdout_results.extend([
            {"Actual": a, "Predicted": p, group_col: str(val)}
            for a, p in zip(y_holdout, y_pred_holdout)
        ])
        train_results.extend([
            {"Actual": a, "Predicted": p, group_col: str(val)}
            for a, p in zip(y_train, y_pred_train)
        ])
        if use_test_split:
            y_pred_test = best_model.predict(X_test)
            test_results.extend([
                {"Actual": a, "Predicted": p, group_col: str(val)}
                for a, p in zip(y_test, y_pred_test)
            ])

    # --- Convert to DataFrames ---
    holdout_df = pd.DataFrame(holdout_results)
    train_df = pd.DataFrame(train_results)
    test_df = pd.DataFrame(test_results) if use_test_split else None

    # --- Sort if group_col is Year ---
    if group_col == "Year":
        holdout_df = holdout_df.sort_values(by="Year").reset_index(drop=True)

    return train_df, test_df, holdout_df

                            # ---- residuals plot ----

def plot_residuals(
    df,
    group_col="Year",
    resid_col="Resid",
    pred_col="Predicted",
    plot_type="box",  # 'box' or 'scatter'
    figsize=(8, 6),
    xlabel=None,
    ylabel=None,
    add_half_std_lines=True,
    half_std_upper=1.365,
    half_std_label= "Half SD" 
):
    """
    Plot residuals as boxplot (by group_col) or scatterplot (vs Predicted).

    Parameters:
    - df: DataFrame containing residuals and predictions
    - group_col: Column for x-axis in boxplot (e.g., 'Year', 'FieldID')
    - resid_col: Name of the residuals column
    - pred_col: Name of predicted values column (used for scatterplot)
    - plot_type: 'box' for boxplot, 'scatter' for scatterplot
    - figsize: Tuple for figure size
    - xlabel: X-axis label
    - ylabel: Y-axis label
    - add_half_std_lines: If True, adds horizontal green dashed lines at ±half_std
    - half_std_upper: Upper threshold (default 1.365)
    - half_std_lower: Lower threshold (default -1.365)
    - half_std_label: Label for horizontal lines
    """

    plt.figure(figsize=figsize)

    if plot_type == "box":
        sns.boxplot(x=group_col, y=resid_col, data=df, palette="Set3")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(group_col)
    elif plot_type == "scatter":
        sns.scatterplot(x=pred_col, y=resid_col, data=df, color="blue", edgecolor="k")
        plt.xlabel("Predicted Values")
    else:
        raise ValueError("plot_type must be either 'box' or 'scatter'")

    plt.axhline(0, linestyle="--", color="red", linewidth=1)

    if add_half_std_lines:
        plt.axhline(half_std_upper, linestyle="--", color="k", linewidth=1, label=f"{half_std_label}")
        
        plt.axhline(-half_std_upper, linestyle="--", color="k", linewidth=1)
        plt.legend()

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)

    plt.tight_layout()
    plt.show()

