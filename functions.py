import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os
import kaggle
from scipy import stats
from sklearn.impute import KNNImputer
import warnings
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import ks_2samp, norm
from statsmodels.stats.proportion import proportions_ztest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
import inspect
import optuna
import lightgbm as lgb
import warnings
import logging
import sys
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
import logging
from sklearn.calibration import cross_val_predict
from sklearn.metrics import log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

def find_outliers_adjusted_iqr(df, column, lower_quantile=0.05, upper_quantile=0.95):
    """Identify outliers using adjusted IQR method."""
    Q1 = df[column].quantile(lower_quantile)
    Q3 = df[column].quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)], lower_bound, upper_bound

def investigate_outliers(df, column, lower_quantile=0.05, upper_quantile=0.95):
    """Investigate outliers in the specified column using given quantiles."""
    outliers, lower_bound, upper_bound = find_outliers_adjusted_iqr(df.dropna(subset=[column]), column, lower_quantile, upper_quantile)
    
    print(f"Statistics for {column}:")
    print(df[column].describe())
    
    valid_labels = outliers['Transported'].value_counts()
    print(f"Label distribution for outliers in {column}:")
    print(valid_labels)
    
    num_affected = len(outliers)
    print(f"Number of affected observations in {column}: {num_affected}")
    
    if not outliers.empty:
        extreme_changes = outliers.sort_values(by=column, key=abs, ascending=False).head(10)
        changes_df = pd.DataFrame({
            'Original': extreme_changes[column],
            'Potential Replacement': np.where(extreme_changes[column] < lower_bound, lower_bound, 
                                              np.where(extreme_changes[column] > upper_bound, upper_bound, extreme_changes[column])),
            'Transported': extreme_changes['Transported']
        })
        print(f"Top 10 most extreme outliers in {column}:\n", changes_df)
    
    x_min, x_max = df[column].min(), df[column].max()
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    axs[0, 0].hist(df[column].dropna(), bins=30, edgecolor='k', alpha=0.7)
    axs[0, 0].set_title(f'Original Distribution of {column}')
    axs[0, 0].set_xlim([x_min, x_max])
    
    capped_values = np.where(df[column] < lower_bound, lower_bound, 
                             np.where(df[column] > upper_bound, upper_bound, df[column]))
    axs[0, 1].hist(capped_values, bins=30, edgecolor='k', alpha=0.7)
    axs[0, 1].set_title(f'Potential Distribution of {column} after Replacement')
    axs[0, 1].set_xlim([x_min, x_max])
    
    axs[1, 0].hist(outliers[column].dropna(), bins=30, edgecolor='k', alpha=0.7, color='r')
    axs[1, 0].set_title(f'Outlier Distribution of {column}')
    axs[1, 0].set_xlim([x_min, x_max])
    
    non_outliers = df[~df.index.isin(outliers.index)][column]
    axs[1, 1].hist(non_outliers.dropna(), bins=30, edgecolor='k', alpha=0.7, color='g')
    axs[1, 1].set_title(f'Non-Outlier Distribution of {column}')
    axs[1, 1].set_xlim([x_min, x_max])
    
    plt.tight_layout()
    plt.show()

def replace_outliers(df, column, lower_quantile=0.05, upper_quantile=0.95):
    """Replace outliers in the specified column using given quantiles and print operation details."""
    outliers, lower_bound, upper_bound = find_outliers_adjusted_iqr(df.dropna(subset=[column]), column, lower_quantile, upper_quantile)
    
    num_affected = len(outliers)
    total_count = df[column].notna().sum()
    percentage_affected = (num_affected / total_count) * 100
    avg_original_value = df[column].mean()
    avg_outlier_value = outliers[column].mean()

    df[column] = np.where(df[column] < lower_bound, lower_bound, 
                          np.where(df[column] > upper_bound, upper_bound, df[column]))
    
    avg_value_after_replacement = df[column].mean()

    print(f"Outlier replacement in '{column}' was successful.")
    print(f"Number of observations affected: {num_affected}")
    print(f"Percentage of total observations affected: {percentage_affected:.2f}%")
    print(f"Average original value: {avg_original_value}")
    print(f"Average outlier value: {avg_outlier_value}")
    print(f"Average value after replacement: {avg_value_after_replacement}")
    
    return df

def draw_bar_plot(dataframe: pd.DataFrame, column: str, title: str, hue: str = None, missing_column: str = None, ax=None):
    """
    Draw a bar plot of the specified column in the dataframe.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the column.
        column (str): The column to plot.
        title (str): The title of the plot.
        hue (str, optional): The column to use for color encoding.
        missing_column (str, optional): The column to check for missing values to add 'missing' indicator.
        ax (matplotlib.axes.Axes, optional): The axis on which to plot. Defaults to None.
    """
    if missing_column:
        dataframe['missing'] = dataframe[missing_column].isna()

    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(x='missing' if missing_column else column, hue=hue, data=dataframe, palette='viridis', alpha=0.7)
        ax.set_title(title)
        total = len(dataframe)
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=8)
        plt.show()
    else:
        sns.countplot(x='missing' if missing_column else column, hue=hue, data=dataframe, palette='viridis', alpha=0.7, ax=ax)
        ax.set_title(title)
        total = len(dataframe)
        for p in ax.patches:
            percentage = '{:.1f}%'.format(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            ax.annotate(percentage, (x, y), ha='center', va='bottom', fontsize=8)



def draw_histogram(dataframe: pd.DataFrame, column: str, title: str, log_scale: bool = False, bins: int = 50, target_feature: str = None, ax=None):
    """
    Draw a histogram of the specified column in the dataframe with optional logarithmic scale,
    custom number of bins, and optional target feature overlay.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the column.
        column (str): The column to plot.
        title (str): The title of the plot.
        log_scale (bool, optional): Whether to use a logarithmic scale. Defaults to False.
        bins (int, optional): Number of bins to use in the histogram. Defaults to 50.
        target_feature (str, optional): The target feature column to overlay True/False or 1/0 distributions. Defaults to None.
        ax (matplotlib.axes.Axes, optional): The axis on which to plot. Defaults to None.
    """
    data = dataframe[column]
    
    if target_feature:
        true_data = dataframe[dataframe[target_feature] == 1][column]
        false_data = dataframe[dataframe[target_feature] == 0][column]
        
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    if target_feature:
        sns.histplot(true_data, bins=bins, color='blue', kde=False, log_scale=(0, log_scale), label='True/1', ax=ax, alpha=0.6)
        sns.histplot(false_data, bins=bins, color='red', kde=False, log_scale=(0, log_scale), label='False/0', ax=ax, alpha=0.6)
        ax.legend()
    else:
        sns.histplot(data, bins=bins, color='skyblue', kde=False, log_scale=(0, log_scale), ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency' if not log_scale else 'Frequency (log scale)')
    
    if ax is None:
        plt.show()




def print_null_statistics(data: pd.DataFrame, column_name: str) -> None:
    """
    Print the count and percentage of missing values in a DataFrame column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column to analyze.

    Returns:
        None
    """
    total_count = len(data[column_name])
    null_count = data[column_name].isnull().sum()
    null_percentage = (null_count / total_count) * 100
    print(f"Feature: {column_name}")
    print(f"Missing values: {null_count} ({null_percentage:.2f}%)")
    print(f"Total: {total_count}")



def normalize_numerical(df: pd.DataFrame, column: str, scaler_type: str='StandardScaler') -> pd.DataFrame:
    """
    Normalize a numerical column in a DataFrame using the specified scaler.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column name to normalize.
        scaler_type (str, optional): The type of scaler to use. Options are 'StandardScaler', 
                                     'MinMaxScaler', 'RobustScaler', 'MaxAbsScaler'. Defaults to 'StandardScaler'.

    Returns:
        pd.DataFrame: The DataFrame with the normalized column.
    """
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"Invalid scaler_type. Expected one of: {list(scalers.keys())}")
    
    scaler = scalers[scaler_type]
    
    imputer = SimpleImputer(strategy='mean')
    df[column] = imputer.fit_transform(df[[column]])
    
    df[[column]] = scaler.fit_transform(df[[column]])
    
    return df




def calculate_imputation_error(true_data, imputed_data, continuous_columns, binary_columns):
    """
    Calculate the imputation error for continuous and binary columns.

    Args:
        true_data (pd.DataFrame): The original DataFrame with missing values.
        imputed_data (pd.DataFrame): The DataFrame after imputation.
        continuous_columns (List[str]): List of column names that are continuous.
        binary_columns (List[str]): List of column names that are binary.
    """
    continuous_error = 0
    binary_error = 0
    
    for column in continuous_columns:
        true_values = true_data[column].dropna()
        imputed_values = imputed_data.loc[true_values.index, column]
        continuous_error += mean_squared_error(true_values, imputed_values)
    
    for column in binary_columns:
        true_values = true_data[column].dropna()
        imputed_values = imputed_data.loc[true_values.index, column]
        binary_error += 1 - accuracy_score(true_values, imputed_values)
    
    return continuous_error, binary_error



def proportion_test(original_data, imputed_data, column):
    """
    Perform a proportion test to compare the proportions of a binary column in the original 
    and imputed datasets.

    Args:
        original_data (pd.DataFrame): The original DataFrame with missing values.
        imputed_data (pd.DataFrame): The DataFrame after imputation.
        column (str): The binary column name to compare.
    """
    original_values = original_data[column].dropna()
    imputed_values = imputed_data.loc[original_values.index, column]
    count = len(original_values)
    
    count_original = original_values.sum()
    count_imputed = imputed_values.sum()
    
    stat, p_value = proportions_ztest([count_original, count_imputed], [count, count])
    return stat, p_value


def ks_test(original_data, imputed_data, column):
    """
    Perform the Kolmogorov-Smirnov test to compare the distributions of a column 
    between the original and imputed datasets.

    Args:
        original_data (pd.DataFrame): The original DataFrame with missing values.
        imputed_data (pd.DataFrame): The DataFrame after imputation.
        column (str): The column name to compare.
    """
    original_values = original_data[column].dropna()
    imputed_values = imputed_data.loc[original_values.index, column]
    stat, p_value = ks_2samp(original_values, imputed_values)
    return stat, p_value



def visualize_imputation(original_data: pd.DataFrame, imputed_data: pd.DataFrame, column: str)-> None:
    """
    Visualize the distribution of a column before and after imputation.

    This function creates side-by-side histograms with KDE plots for the original
    and imputed data of the specified column.

    Args:
        original_data (pd.DataFrame): The original DataFrame with missing values.
        imputed_data (pd.DataFrame): The DataFrame after imputation.
        column (str): The column name to visualize.
    """
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(original_data[column].dropna(), kde=True, color='blue')
    plt.title(f'Original {column} Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(imputed_data[column], kde=True, color='green')
    plt.title(f'Imputed {column} Distribution')
    plt.show()



def cross_validate(model, X, y, cv=5):
    """
    Perform cross-validation for a given model.
    
    Parameters:
    model (object): Model object (e.g., XGBClassifier, LGBMClassifier, etc.).
    X (pd.DataFrame or np.array): Feature matrix.
    y (pd.Series or np.array): Target vector.
    cv (int): Number of cross-validation folds. Default is 5.
    
    Returns:
    dict: Dictionary containing cross-validated accuracy and logloss.
    """

    cv_predictions = cross_val_predict(model, X, y, cv=cv, method='predict')
    cv_probabilities = cross_val_predict(model, X, y, cv=cv, method='predict_proba')

    accuracy = accuracy_score(y, cv_predictions)
    logloss = log_loss(y, cv_probabilities)

    print(f"Model: {type(model).__name__}")
    print(f"Cross-validated Accuracy: {accuracy:.4f}")
    print(f"Cross-validated Logloss: {logloss:.4f}")

    return {"model_name": type(model).__name__, "accuracy": accuracy, "logloss": logloss}

def suggest_hyperparameters(trial, X, y, model_type):
    """
    Suggest hyperparameters for LightGBM, XGBoost, or CatBoost.
    
    Parameters:
    trial (optuna.trial): Optuna trial object.
    X (pd.DataFrame or np.array): Feature matrix.
    y (pd.Series or np.array): Target vector.
    model_type (str): Type of model ('lightgbm', 'xgboost', 'catboost').
    
    Returns:
    float: Cross-validated accuracy (or another metric if you choose).
    """
    
    valid_models = ['lightgbm', 'xgboost', 'catboost']

    if model_type not in valid_models:
        raise ValueError(f"Invalid model type '{model_type}'. Please choose from {valid_models}.")

    if model_type == 'lightgbm':
        param = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 35),
            'max_depth': trial.suggest_int('max_depth', 12, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 6),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 0.01, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 0.02, log=True),
            'verbosity': -1
        }
        model = lgb.LGBMClassifier(**param, random_state=42, n_jobs=-1)
    
    elif model_type == 'xgboost':
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'gamma': trial.suggest_float('gamma', 0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'scale_pos_weight': trial.suggest_int('scale_pos_weight', 1, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
            'n_estimators': trial.suggest_int('n_estimators', 100, 300),
            'verbosity': 0
        }
        model = xgb.XGBClassifier(**param, random_state=42, n_jobs=-1)
    
    elif model_type == 'catboost':
        param = {
            'depth': trial.suggest_int('depth', 6, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.02, log=True),
            'iterations': trial.suggest_int('iterations', 100, 300),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-5, 0.01, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'rsm': trial.suggest_float('rsm', 0.8, 1.0),
            'iterations': trial.suggest_int('iterations', 50, 200),
            'early_stopping_rounds': 50,
            'verbose': 0
        }
        model = cb.CatBoostClassifier(**param, random_state=42)

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        results = cross_validate(model, X, y, cv=5)
    
    print(f"[INFO] {model_type.capitalize()} model: mean cross-validated accuracy = {results['accuracy']:.4f}")
    
    return results['accuracy']


def preprocess_and_impute_data_test(df):
    # Extract PassengerId and preserve it as a separate variable
    passenger_ids = df['PassengerId'].copy()
    df = df.drop(columns=['PassengerId'])


    # Replace boolean
    df['CryoSleep'] = df['CryoSleep'].replace({True: 1, False: 0}).astype(float)
    df['VIP'] = df['VIP'].replace({True: 1, False: 0}).astype(float)
    
    # Encode categorical
    homeplanet_dummies = pd.get_dummies(df['HomePlanet'], prefix='HomePlanet')
    homeplanet_dummies[df['HomePlanet'].isna()] = float('nan')
    destination_dummies = pd.get_dummies(df['Destination'], prefix='Destination')
    destination_dummies[df['Destination'].isna()] = float('nan')
    df = pd.concat([df, homeplanet_dummies, destination_dummies], axis=1)
    df.drop(['HomePlanet', 'Destination'], axis=1, inplace=True)

    # Cabin
    df[['deck', 'room_number', 'side']] = df['Cabin'].str.split('/', expand=True)
    df['deck'] = df['deck'].where(df['Cabin'].notna(), None)
    df['room_number'] = df['room_number'].where(df['Cabin'].notna(), None)
    df['side'] = df['side'].where(df['Cabin'].notna(), None)
    df['port'] = df['side'].apply(lambda x: 1.0 if x == 'P' else (0.0 if x == 'S' else float('nan')))
    df['starboard'] = df['side'].apply(lambda x: 1.0 if x == 'S' else (0.0 if x == 'P' else float('nan')))
    deck_dummies = pd.get_dummies(df['deck'], prefix='deck').astype(float)
    deck_dummies[df['deck'].isna()] = float('nan')
    df = pd.concat([df, deck_dummies], axis=1)
    df.drop(['Cabin', 'deck', 'side'], axis=1, inplace=True)
    df['room_number'] = pd.to_numeric(df['room_number'], errors='coerce')

    # Drop Name column
    df.drop(columns=['Name'], inplace=True)

    # Normalization of numerical columns
    numerical_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for column in numerical_columns:
        df = normalize_numerical(df, column, scaler_type='RobustScaler')

    # Find the optimal k using cross-validation
    continuous_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'room_number']
    binary_columns = ['CryoSleep', 'VIP', 'HomePlanet_Earth', 'HomePlanet_Europa', 'HomePlanet_Mars', 
                      'Destination_55 Cancri e', 'Destination_PSO J318.5-22', 'port', 'starboard', 
                      'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'deck_T']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_values = range(1, 11)
    errors = []

    for k in k_values:
        continuous_error_sum = 0
        binary_error_sum = 0
        for train_index, test_index in kf.split(df):
            train_data, test_data = df.iloc[train_index], df.iloc[test_index]
            
            imputer = KNNImputer(n_neighbors=k)
            imputer.fit(train_data)
            
            imputed_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns, index=test_data.index)
            
            continuous_error, binary_error = calculate_imputation_error(test_data, imputed_data, continuous_columns, binary_columns)
            continuous_error_sum += continuous_error
            binary_error_sum += binary_error
        
        errors.append((k, continuous_error_sum / kf.get_n_splits(), binary_error_sum / kf.get_n_splits()))

    errors_df = pd.DataFrame(errors, columns=['k', 'continuous_error', 'binary_error'])
    optimal_k = errors_df.loc[errors_df['continuous_error'].idxmin(), 'k']

    # Apply kNN Imputation
    final_imputer = KNNImputer(n_neighbors=int(optimal_k))
    df_imputed = pd.DataFrame(final_imputer.fit_transform(df), columns=df.columns)
    df_imputed['PassengerId'] = passenger_ids

    return df_imputed



def create_comparison_table(results_list, metric='accuracy'):
    """
    Create a comparison table showing the performance of models on full and reduced datasets.

    This function generates a DataFrame comparing the specified metric between full and reduced
    models, and calculates the absolute difference between them.

    Args:
        results_list (List[Dict[str, any]]): A list of result dictionaries, each containing model performance metrics.
        metric (str, optional): The metric to compare. Defaults to 'accuracy'.
    """
    local_vars = inspect.currentframe().f_back.f_locals
    data = {'Model': [], 'Full': [], 'Reduced': [], 'Difference': []}
    
    for result in results_list:
        var_name = [name for name, value in local_vars.items() if value is result][0]
        model_name = result['model_name']
        value = result[metric]
        
        if model_name not in data['Model']:
            data['Model'].append(model_name)
            data['Full'].append(None)
            data['Reduced'].append(None)
            data['Difference'].append(None)
        
        index = data['Model'].index(model_name)
        
        if 'reduced' in var_name.lower():
            data['Reduced'][index] = value
        else:
            data['Full'][index] = value
            
        if data['Full'][index] is not None and data['Reduced'][index] is not None:
            data['Difference'][index] = abs(data['Full'][index] - data['Reduced'][index])
    
    df = pd.DataFrame(data)
    return df
