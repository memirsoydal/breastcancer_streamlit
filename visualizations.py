# visualizations.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def hist_grid(data, columns, hue=None, rows=2, cols=5, figsize=(20, 20), suptitle="Distribution of Columns"):
    """
    Plots a 2x5 grid of histogram distributions of given columns.

    :param cols:
    :param rows:
    :param data: Pandas DataFrame containing the data to plot
    :param columns: List of column names to plot
    :param hue: Optional. Column name to be used as hue in plots
    :param figsize: Tuple indicating figure size. Default is (20, 10)
    :param suptitle: Title for the entire figure. Default is "Distribution of Columns"
    """
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    # Flatten the array of axes for easy iterating
    axs = axs.flatten()
    # Loop through the columns and create a plot for each
    for index, col in enumerate(columns):
        sns.histplot(data=data, x=col, kde=True, hue=hue, ax=axs[index], bins=20)
        axs[index].set_title(col)
    fig.suptitle(suptitle)
    plt.tight_layout()
    return fig


# Function to read the uploaded file
def read_file(uploaded_file):
    df = pd.read_csv(uploaded_file, index_col='id')
    df['diagnosis'] = df['diagnosis'].replace({"B": 0, "M": 1}).astype("category")
    return df


def log_transform(x):
    return np.log(x + 1)


# Function for data preprocessing
def preprocess_data(df):
    X = df.drop(['diagnosis', 'Unnamed: 32'], axis=1)
    y = df['diagnosis']
    # Split the data first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the preprocessing pipeline
    preprocess_pipeline = ColumnTransformer(transformers=[
        ('log_scale', Pipeline(steps=[
            ('log', FunctionTransformer(log_transform)),
            ('scale', StandardScaler())
        ]), X.columns)
    ], remainder='passthrough')

    # Fit and transform the training data, then reconstruct DataFrame
    X_train_transformed = preprocess_pipeline.fit_transform(X_train)
    X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=X_train.columns, index=X_train.index)

    # Transform the test data, then reconstruct DataFrame
    X_test_transformed = preprocess_pipeline.transform(X_test)
    X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=X_test.columns, index=X_test.index)

    return X_train_transformed_df, X_test_transformed_df, y_train, y_test


# Function for model training and prediction
def train_predict(model_choice, X_train, X_test, y_train):
    model_params = get_model_params(model_choice)
    fold = StratifiedKFold(n_splits=5)
    model = GridSearchCV(model_params['model'], model_params['params'], scoring='f1', cv=fold, return_train_score=False)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, predictions


# Function to return model parameters
def get_model_params(model_choice):
    model_params = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            }
        },
        # Assuming this is part of your model_params definition
        'SVC': {
            'model': SVC(probability=True, random_state=42),  # Enable probability estimation
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'params': {}
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.01, 0.1, 1],
                'tol': [1e-6, 1e-3]
            }
        }
    }
    return model_params[model_choice]


# Function to plot confusion matrix
def plot_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, square=True, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_yticklabels(['Negative', 'Positive'], rotation=0)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    st.pyplot(fig)


def apply_feature_selection(X_train, X_test, y_train, method):
    if method == "LDA":
        lda = LDA(n_components=min(len(np.unique(y_train)) - 1, X_train.shape[1]))
        X_train = lda.fit_transform(X_train, y_train)
        X_test = lda.transform(X_test)
    elif method == "RFE":
        rfe = RFE(estimator=LogisticRegression())
        X_train = rfe.fit_transform(X_train, y_train)
        X_test = rfe.transform(X_test)
    # No need for an explicit "None" case, as we return the data unchanged
    return X_train, X_test


def plot_with_pca(X_test, y_test, predictions):
    # Apply PCA
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test)

    test_df = pd.DataFrame(X_test_pca, columns=['Principal Component 1', 'Principal Component 2'])
    test_df['ID'] = y_test.index
    test_df['True Label'] = y_test.reset_index(drop=True)
    test_df['Prediction'] = predictions
    test_df['Correct'] = test_df['True Label'] == test_df['Prediction']

    # Determine TP, TN, FP, FN for coloring
    test_df['Type'] = np.where((test_df['True Label'] == 1) & (test_df['Prediction'] == 1), 'TP',
                               np.where((test_df['True Label'] == 0) & (test_df['Prediction'] == 0), 'TN',
                                        np.where((test_df['True Label'] == 1) & (test_df['Prediction'] == 0), 'FN',
                                                 'FP')))

    # Plot using Plotly
    fig = px.scatter(test_df, x='Principal Component 1', y='Principal Component 2', color='Type',
                     hover_data=['True Label', 'Prediction', 'ID'], title='PCA of Test Data by Classification Type')
    st.plotly_chart(fig, use_container_width=True)


def plot_diagnosis_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='diagnosis', data=df)
    plt.title('Distribution of Diagnosis Classes')
    plt.xlabel('Diagnosis')
    plt.ylabel('Count')
    st.pyplot(plt)


def plot_missing_values(df):
    # Calculate the number of missing values per column
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]  # Filter columns with missing values
    missing_values.sort_values(inplace=True)

    # Plot
    plt.figure(figsize=(10, 6))
    missing_values.plot(kind='bar')
    plt.title('Number of Missing Values Per Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Values Count')
    plt.xticks(rotation=45, ha="right")
    st.pyplot(plt)


def plot_learning_curve(estimator, X, y, title="Learning Curve", ylim=None, cv=None, n_jobs=-1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    fig, ax = plt.subplots(figsize=(8, 6))  # You can adjust the figure size as needed

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.set_title(title)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid()

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1, color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    ax.legend(loc="best")
    st.pyplot(fig)


def classification_report_to_dataframe(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df['support'] = df['support'].astype(int)  # Convert support from float to int
    return df


def plot_correlation_heatmaps(data, mean_columns, se_columns, worst_columns):
    # Calculate correlation matrices
    mean_columns_corr = data[['diagnosis'] + mean_columns].corr()
    se_columns_corr = data[['diagnosis'] + se_columns].corr()
    worst_columns_corr = data[['diagnosis'] + worst_columns].corr()

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 20))
    sns.heatmap(mean_columns_corr, annot=True, cmap="cividis", ax=axes[0], cbar_kws={'shrink': .5})
    sns.heatmap(se_columns_corr, annot=True, cmap="cividis", ax=axes[1], cbar_kws={'shrink': .5})
    sns.heatmap(worst_columns_corr, annot=True, cmap="cividis", ax=axes[2], cbar_kws={'shrink': .5})

    # Set titles for subplots
    axes[0].set_title('Mean Features Correlation')
    axes[1].set_title('SE Features Correlation')
    axes[2].set_title('Worst Features Correlation')

    plt.tight_layout()
    st.pyplot(fig)
