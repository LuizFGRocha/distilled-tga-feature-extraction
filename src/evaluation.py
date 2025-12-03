import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold, LeaveOneOut
from sklearn.metrics import r2_score

def evaluate_with_bootstrap(encodings, labels, n_bootstraps=1000, alpha=0.05):
    """
    Evaluates encoding quality using bootstrapping to estimate the R^2 score distribution.
    """
    n_samples = len(encodings)
    bootstrap_scores = []
    
    for _ in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot, y_boot = encodings[indices], labels[indices]
        
        # Out-of-bag samples for validation
        oob_indices = np.array(list(set(range(n_samples)) - set(indices)))
        if len(oob_indices) == 0:
            continue 
            
        X_oob, y_oob = encodings[oob_indices], labels[oob_indices]

        regressor = RandomForestRegressor(n_estimators=100, random_state=None, n_jobs=-1)
        regressor.fit(X_boot, y_boot)
        score = regressor.score(X_oob, y_oob)
        bootstrap_scores.append(score)
        
    lower_bound = np.percentile(bootstrap_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    mean_score = np.mean(bootstrap_scores)
    
    return mean_score, lower_bound, upper_bound

def evaluate_with_cv(encodings, labels, n_splits=2, n_repeats=20, alpha=0.05):
    """
    Evaluates encoding quality using Repeated K-Fold Cross-Validation.
    """
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    cv_scores = []

    for train_idx, test_idx in cv.split(encodings):
        X_train, X_test = encodings[train_idx], encodings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        regressor.fit(X_train, y_train)

        score = regressor.score(X_test, y_test)
        cv_scores.append(score)

    mean_score = np.mean(cv_scores)
    lower_bound = np.percentile(cv_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(cv_scores, 100 * (1 - alpha / 2))

    return mean_score, lower_bound, upper_bound

def evaluate_with_loo(encodings, labels, n_bootstraps=1000, alpha=0.05):
    """
    Evaluates encoding quality using Leave-One-Out Cross-Validation.
    Returns the R2 score calculated on the aggregated predictions (Q2).
    Confidence intervals are computed via bootstrap resampling of the residuals.
    """
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(encodings):
        X_train, X_test = encodings[train_idx], encodings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        regressor.fit(X_train, y_train)
        
        prediction = regressor.predict(X_test)
        
        y_true.append(y_test[0])
        y_pred.append(prediction[0])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    score = r2_score(y_true, y_pred)
    
    n_samples = len(y_true)
    bootstrap_r2_scores = []
    
    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if np.var(y_true_boot) > 0:
            r2_boot = r2_score(y_true_boot, y_pred_boot)
            bootstrap_r2_scores.append(r2_boot)
    
    lower_bound = np.percentile(bootstrap_r2_scores, 100 * (alpha / 2))
    upper_bound = np.percentile(bootstrap_r2_scores, 100 * (1 - alpha / 2))
    
    return score, lower_bound, upper_bound
