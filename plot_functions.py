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
    return 100.0 * np.sum(observed - predicted) / np.sum(observed)

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


def run_xgb_kfold(X, y, n_splits=5, random_state=153, params=None):
    """
    Run XGBoost regression with KFold CV.
    
    Returns:
        results_df: DataFrame with columns:
                    ['Fold', 'Set', 'Actual', 'Predicted']
        metrics_df: DataFrame with MAE and R2 per fold
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

        # Save results
        results.extend([
            pd.DataFrame({
                'Fold': fold,
                'Set': 'Train',
                'Actual': y_train,
                'Predicted': y_train_pred
            }),
            pd.DataFrame({
                'Fold': fold,
                'Set': 'Test',
                'Actual': y_test,
                'Predicted': y_test_pred
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