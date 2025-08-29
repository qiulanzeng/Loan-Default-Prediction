from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from tqdm import tqdm
import joblib
from joblib import parallel_backend

class TqdmJoblib(joblib.Parallel):
    def __init__(self, total=None, *args, **kwargs):
        self._total = total
        self._pbar = tqdm(total=total, desc="GridSearchCV Progress")
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with self._pbar:
            return super().__call__(*args, **kwargs)

    def print_progress(self):
        if self._pbar:
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()


class ModelTrainer:
    def __init__(self, config=None):
        self.config = config

    def train(self, preprocessor, X_train, y_train):
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])

        param_grid = {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [None, 10],
            'classifier__min_samples_split': [2],
            'classifier__min_samples_leaf': [1]
        }

        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        total_fits = len(ParameterGrid(param_grid)) * cv.get_n_splits()

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=cv,
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )

        with parallel_backend('loky'):
            grid_search._parallel_backend = TqdmJoblib(n_jobs=-1, total=total_fits)
            grid_search.fit(X_train, y_train)

        joblib.dump(grid_search.best_estimator_, 'loan_default_model_pipeline.pkl')
        print("Saved model to 'loan_default_model_pipeline.pkl'")

        return grid_search.best_estimator_
