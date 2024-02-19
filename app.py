from visualizations import *


def main():
    st.title('Breast Cancer Diagnostic Prediction')

    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    model_choice = st.sidebar.selectbox("Choose the ML model", ("KNN", "SVC", "Naive Bayes",
                                                                "Logistic Regression"))
    feature_selection_method = st.sidebar.selectbox("Feature Selection Method", ("None", "LDA", "RFE"))

    if uploaded_file is not None:
        df = read_file(uploaded_file)
        X_train, X_test, y_train, y_test = preprocess_data(df)

        tab1, tab2, tab3 = st.tabs(["EDA and Visualization", "Preprocessing", "Performance"])

        with tab1:
            st.header("EDA and Visualization")
            st.write("""In this section, we explore the dataset to understand its characteristics, distribution of 
            variables, and the relationship between different features. The goal is to gain insights that could be 
            useful for data preprocessing and model building.""")
            st.write(df.head(10))

            st.subheader("Missing Values")
            st.write("""We can plot the amount of missing values in our dataset. From this plot, we can easily see 
            'Unnamed: 32' column does not provide any information and must be dropped.""")
            plot_missing_values(df)

            # Header for Distribution Analysis
            st.subheader("Target Distribution Analysis")
            st.write("""We can check the distribution of the target feature. First thing we notice is that there are 
            more benign cases than malignant cases. This could suggest a StratifiedKFold method to keep the folds 
            balanced.""")
            plot_diagnosis_distribution(df)

            st.subheader("Histogram Analysis")
            st.write("""From these plots we can see, malignant cases tend to be more separated, and having larger 
            values than benign cases. We can also see our features are in different ranges. This might suggest a 
            scaling. We can easily see there's a great skewness in these columns, so as mean columns but these are on 
            an extreme level. We might need to apply log transformation to get a better understanding of these 
            columns.""")
            hist_fig = hist_grid(df, df.columns[1:-1], hue='diagnosis', rows=6, cols=5,
                                 suptitle="Distribution of Columns")
            st.pyplot(hist_fig)

            st.subheader("Correlation Heatmaps")
            st.write("""We can check these columns to see if they have any correlation with the target value or 
            within each other. From this heatmap we can see that some columns show very high correlation. This could 
            be due to the fact that many of these columns went through feature engineering, and they are related to 
            each other in some way. SE Columns seem to show very low correlation values, this might suggest Feature 
            Selecting for reducing dimensionality and complexity.""")
            mean_columns = df.columns[1:11].to_list()
            se_columns = df.columns[11:21].to_list()
            worst_columns = df.columns[21:-1].to_list()

            # Plot heatmaps
            plot_correlation_heatmaps(df.iloc[:, 0:-1], mean_columns, se_columns, worst_columns)

        with tab2:
            st.header("Preprocessing Steps")
            st.write("""From EDA, we have discovered our data needs dropping, scaling and log transformation. We'll 
            create a pipeline to apply these steps in an organized fashion.""")

            st.write("### Original Data")
            st.write(df.tail(10))
            st.write("### Transformed Data")

            X_train_df = pd.DataFrame(X_train, columns=df.drop(['diagnosis', 'Unnamed: 32'], axis=1).columns)
            st.write(X_train_df.tail(10))

            st.subheader("Feature Selection")
            st.write("""Since our dataset contains high number of features, reducing dimensionality could help models 
            learn better. To do this we can apply RFE, which recursively selects the most important features and 
            creates a subset of features for the given size. We can also use LDA which tries to maximize the variance 
            between the two classes we have.""")
        with tab3:
            st.header("Model Performance")

            X_train_fs, X_test_fs = apply_feature_selection(X_train, X_test, y_train, feature_selection_method)
            model, predictions = train_predict(model_choice, X_train_fs, X_test_fs, y_train)
            st.write("We will evaluate the model's performance with many metrics and finally analyse which cases our "
                     "model have missed.")
            st.write(f"Best Parameters: {model.best_params_}")
            st.subheader("Learning Curve")
            st.write("""We can see plot learning curve to see how the model reacts to different sample sizes.""")
            plot_learning_curve(model, X_train_fs, y_train, cv=5)

            st.subheader('Classification Report')
            df_report = classification_report_to_dataframe(y_test, predictions)
            st.dataframe(df_report, use_container_width=True)

            st.subheader("Confusion Matrix")
            plot_confusion_matrix(y_test, predictions)

            if feature_selection_method != 'LDA':
                plot_with_pca(X_test_fs, y_test, predictions)
    else:
        st.write("Please upload a CSV file to proceed.")


if __name__ == "__main__":
    main()
