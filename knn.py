# Load the preprocessed data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

def clean_data(X):
    """Clean data by handling NaN, infinite values, and extreme outliers"""
    # Replace infinite values with NaN
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    
    # For each column, replace NaN with median for numerical columns
    for column in X_clean.columns:
        if X_clean[column].dtype in ['int64', 'float64']:
            median_val = X_clean[column].median()
            X_clean[column] = X_clean[column].fillna(median_val)
            
            # Handle extreme outliers (optional)
            # q1 = X_clean[column].quantile(0.01)
            # q3 = X_clean[column].quantile(0.99)
            # X_clean[column] = X_clean[column].clip(q1, q3)
    
    return X_clean

def train_knn_model(X_train, X_test, y_train, y_test):
    # Initialize timer
    start_time = time.time()
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_neighbors': [5, 7, 9, 11, 13],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # Initialize base KNN model
    base_knn = KNeighborsClassifier()
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        base_knn,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_knn = grid_search.best_estimator_
    training_time = time.time() - start_time
    
    # Make predictions
    y_pred = best_knn.predict(X_test)
    y_prob = best_knn.predict_proba(X_test)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get cross-validation scores
    cv_scores = cross_val_score(best_knn, X_train, y_train, cv=5)
    
    # Compile detailed metrics
    detailed_metrics = {
        'model_type': 'KNN',
        'best_parameters': grid_search.best_params_,
        'accuracy': cv_scores.mean(),
        'std_dev': cv_scores.std(),
        'training_time': training_time,
        'confusion_matrix': conf_matrix,
        'confusion_matrix_normalized': conf_matrix_normalized,
        'classification_report': class_report,
        'cv_scores': cv_scores,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    return best_knn, detailed_metrics

# Train the model and get metrics
best_knn_model, knn_metrics = train_knn_model(X_train, X_test, y_train, y_test)

# Save detailed metrics
with open('knn_detailed_metrics.pkl', 'wb') as f:
    pickle.dump(knn_metrics, f)

# Save the model
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(best_knn_model, f)

# Print results
print("\nKNN Model Results")
print("-" * 50)
print(f"\nBest Parameters: {knn_metrics['best_parameters']}")
print(f"\nTraining time: {knn_metrics['training_time']:.2f} seconds")
print("\nClassification Report:")
print(classification_report(y_test, best_knn_model.predict(X_test)))
print("\nConfusion Matrix:")
print(knn_metrics['confusion_matrix'])
print("\nCross-validation scores:", knn_metrics['cv_scores'])
print(f"Average CV score: {knn_metrics['accuracy']:.3f} (+/- {knn_metrics['std_dev'] * 2:.3f})")