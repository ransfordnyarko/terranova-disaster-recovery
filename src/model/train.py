import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dotenv import load_dotenv
import os
from pathlib import Path
import optuna
from optuna.samplers import TPESampler


load_dotenv()

EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "Default Experiment")
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR.parent.parent / "data"
ENGINEERED_DATA_DIR = DATA_DIR / "engineered"

# Clean feature definitions — no leakage
NUMERICAL_FEATURES = [
    'fyDeclared',
    'designatedArea_count',
    'ihProgramDeclared',
    'iaProgramDeclared',
    'paProgramDeclared',
    'hmProgramDeclared',
    'incident_duration_days',
    'is_ongoing',
    'declaration_month',
    'declaration_quarter',          # target encoded state
    'historical_median_cost',
    'historical_project_count',
    'historical_large_project_count',
    'historical_median_ihp',
    'historical_median_ha',
    'historical_median_ona',
    'historical_median_pa',
    'historical_median_hmgp',
    'historical_median_catab',
    'historical_median_catc2g',
    'programs_activated',
    'is_low_program_activation',
    'is_emergency_declaration',
    'area_bin_score',
    'area_bin_score_dr',
    'area_x_dr',
    'historical_duration_days',
    'is_biological_em',
]

CATEGORICAL_FEATURES = [
    'incidentType',
    'declarationType',
]

DATE_FEATURES = [
    'declarationDate',
    'incidentBeginDate',
]

TARGET = 'total_disaster_cost'
LOG_TARGET = 'log_total_cost'

TUNABLE_MODELS = ['random_forest', 'gradient_boosting', 'ridge', 'lasso']


class ModelTrainer:
    def __init__(self, n_trials=50):
        mlflow.set_experiment(EXPERIMENT_NAME)
        self.n_trials = n_trials
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
        }
        self.results = {}
        self.best_params = {}

    def load_data(self):
        """Load engineered dataset."""
        df = pd.read_csv(ENGINEERED_DATA_DIR / "disaster_costs.csv")
        print(f"Loaded data shape: {df.shape}")
        print(f"Numerical features: {df.select_dtypes(include=[np.number]).columns.tolist()}")
        print(f"Categorical features: {df.select_dtypes(include=[object]).columns.tolist()}")
        return df
    
    def _extract_declaration_month(self, X):
        return pd.to_datetime(X['declarationDate'], utc=True).dt.month.values.reshape(-1, 1)

    def _extract_declaration_quarter(self, X):
        return pd.to_datetime(X['declarationDate'], utc=True).dt.quarter.values.reshape(-1, 1)

    def _extract_incident_month(self, X):
        return pd.to_datetime(X['incidentBeginDate'], utc=True).dt.month.values.reshape(-1, 1)

    def _extract_incident_quarter(self, X):
        return pd.to_datetime(X['incidentBeginDate'], utc=True).dt.quarter.values.reshape(-1, 1)

    def prepare_features(self, df):
        """Separate features and target. Log transform target."""
        leakage_cols = [
            'total_obligated', 'federal_obligated', 'avg_project_cost',
            'applicant_count', 'project_count', 'large_project_count',
            'small_project_count', 'disasterNumber', 'incidentEndDate'
        ]
        found_leakage = [c for c in leakage_cols if c in df.columns]
        if found_leakage:
            print(f"WARNING — dropping leakage columns: {found_leakage}")
            df = df.drop(columns=found_leakage)

        # Diagnose before dropping
        print(f"\nNull target values: {df[TARGET].isna().sum()}")
        print(f"Zero target values: {(df[TARGET] == 0).sum()}")
        print(f"Negative target values: {(df[TARGET] < 0).sum()}")

        # Drop rows where target is null — can't train without a label
        df = df.dropna(subset=[TARGET])

        # Negative costs are data errors — clip to 0
        df[TARGET] = df[TARGET].clip(lower=0)

        # Log transform — log1p handles zeros safely
        df[LOG_TARGET] = np.log1p(df[TARGET])

        # Verify target is clean
        assert df[LOG_TARGET].isna().sum() == 0, "NaN in log target after cleaning"
        assert np.isinf(df[LOG_TARGET]).sum() == 0, "Inf in log target after cleaning"

        print(f"Rows after dropping null targets: {len(df)}")

        missing_num = [f for f in NUMERICAL_FEATURES if f not in df.columns]
        missing_cat = [f for f in CATEGORICAL_FEATURES if f not in df.columns]
        missing_date = [f for f in DATE_FEATURES if f not in df.columns]

        if missing_num:
            print(f"WARNING — missing numerical features: {missing_num}")
        if missing_cat:
            print(f"WARNING — missing categorical features: {missing_cat}")
        if missing_date:
            print(f"WARNING — missing date features: {missing_date}")

        available_num = [f for f in NUMERICAL_FEATURES if f in df.columns]
        available_cat = [f for f in CATEGORICAL_FEATURES if f in df.columns]
        available_date = [f for f in DATE_FEATURES if f in df.columns]

        X = df[available_num + available_cat + available_date]
        y = df[LOG_TARGET]

        # Final null check across all features
        null_features = X.isna().sum()
        null_features = null_features[null_features > 0]
        if len(null_features) > 0:
            print(f"\nWARNING — null values in features:\n{null_features}")

        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target distribution:\n{y.describe()}")

        return X, y, available_num, available_cat, available_date

    def build_preprocessor(self, numerical_features, categorical_features, date_features=None):
        """Build column transformer pipeline."""
        transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(
                    drop='first',
                    handle_unknown='ignore',  # handles unseen categories at inference
                    sparse_output=False
                ), categorical_features),
        ]
        
        if 'declarationDate' in date_features:
            transformers += [
                ('decl_month', FunctionTransformer(self._extract_declaration_month), date_features),
                ('decl_quarter', FunctionTransformer(self._extract_declaration_quarter), date_features),
            ]

        if 'incidentBeginDate' in date_features:
            transformers += [
                ('inc_month', FunctionTransformer(self._extract_incident_month), date_features),
                ('inc_quarter', FunctionTransformer(self._extract_incident_quarter), date_features)
            ]

        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop') 
        return preprocessor

    def train_and_log(self, 
                      model_name, 
                      model, 
                      X_train, 
                      X_test, 
                      y_train, 
                      y_test,
                      numerical_features, 
                      categorical_features,
                      date_features,
                      tuned=False
                      ):
        """Train a single model and log everything to MLflow."""
        with mlflow.start_run(run_name=model_name):

            # Build pipeline
            pipeline = Pipeline([
                ('preprocessor', self.build_preprocessor(
                    numerical_features, categorical_features, date_features
                )),
                ('model', model)
            ])

            # Cross validation on training set
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=kf, scoring='r2', n_jobs=-1
            )

            # Fit on full training set
            pipeline.fit(X_train, y_train)

            # Evaluate on test set
            metrics = self.evaluate(pipeline, X_test, y_test)

            # Log params
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("tuned", tuned)
            mlflow.log_param("numerical_features", numerical_features)
            mlflow.log_param("categorical_features", categorical_features)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))

            if tuned and model_name in self.best_params:
                mlflow.log_params(self.best_params[model_name])

            # Log metrics
            mlflow.log_metric("cv_r2_mean", cv_scores.mean())
            mlflow.log_metric("cv_r2_std", cv_scores.std())
            mlflow.log_metric("test_r2", metrics['r2'])
            mlflow.log_metric("test_rmse", metrics['rmse'])
            mlflow.log_metric("test_mae", metrics['mae'])
            mlflow.log_metric("test_rmse_dollars", metrics['rmse_dollars'])
            mlflow.log_metric("test_mae_dollars", metrics['mae_dollars'])

            # Log model
            mlflow.sklearn.log_model(pipeline, artifact_path=model_name)

            print(f"\n{'='*40}")
            print(f"Model: {model_name}")
            print(f"CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"Test R²  : {metrics['r2']:.4f}")
            print(f"Test RMSE: ${metrics['rmse_dollars']:,.0f}")
            print(f"Test MAE : ${metrics['mae_dollars']:,.0f}")

            self.results[model_name] = {
                'pipeline': pipeline,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                **metrics
            }

        return pipeline

    def evaluate(self, pipeline, X_test, y_test):
        """Evaluate model on test set — report both log scale and dollar scale."""
        y_pred_log = pipeline.predict(X_test)

        # Log scale metrics
        r2 = r2_score(y_test, y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_log))
        mae = mean_absolute_error(y_test, y_pred_log)

        # Convert back to dollar scale for interpretability
        y_pred_dollars = np.expm1(y_pred_log)
        y_test_dollars = np.expm1(y_test)
        rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
        mae_dollars = mean_absolute_error(y_test_dollars, y_pred_dollars)

        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'rmse_dollars': rmse_dollars,
            'mae_dollars': mae_dollars
        }

    def summarize_results(self):
        """Print ranked model comparison."""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON — ranked by CV R²")
        print(f"{'='*60}")

        summary = pd.DataFrame({
            name: {
                'CV R² (mean)': f"{res['cv_r2_mean']:.4f}",
                'CV R² (std)': f"±{res['cv_r2_std']:.4f}",
                'Test R²': f"{res['r2']:.4f}",
                'Test RMSE ($)': f"${res['rmse_dollars']:,.0f}",
                'Test MAE ($)': f"${res['mae_dollars']:,.0f}",
            }
            for name, res in self.results.items()
        }).T.sort_values('CV R² (mean)', ascending=False)

        print(summary.to_string())

        best = max(self.results, key=lambda x: self.results[x]['cv_r2_mean'])
        print(f"\nBest model: {best}")
        return best
    
    def _get_optuna_params(self, trial, model_name):
        """
        Define hyperparameter search space per model.
        Only called for tunable models.
        """
        if model_name == 'random_forest':
            return {
                'model__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'model__max_depth': trial.suggest_int('max_depth', 3, 20),
                'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'model__max_features': trial.suggest_categorical(
                    'max_features', ['sqrt', 'log2', None]
                ),
            }

        elif model_name == 'gradient_boosting':
            return {
                'model__n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'model__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'model__max_depth': trial.suggest_int('max_depth', 2, 8),
                'model__min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'model__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'model__subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'model__max_features': trial.suggest_categorical(
                    'max_features', ['sqrt', 'log2', None]
                ),
            }

        elif model_name == 'ridge':
            return {
                'model__alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            }

        elif model_name == 'lasso':
            return {
                'model__alpha': trial.suggest_float('alpha', 1e-3, 1e3, log=True),
            }

    def _tune_model(self, model_name, model, X_train, y_train,
                    numerical_features, categorical_features, date_features):
        """
        Run Optuna hyperparameter search for a single model.
        Optimizes CV R² on training set only — test set never touched.
        """
        print(f"\nTuning {model_name} with {self.n_trials} trials...")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        def objective(trial):
            params = self._get_optuna_params(trial, model_name)

            pipeline = Pipeline([
                ('preprocessor', self.build_preprocessor(
                    numerical_features, categorical_features, date_features
                )),
                ('model', model.__class__(random_state=42, n_jobs=-1)
                 if hasattr(model, 'n_jobs')
                 else model.__class__(random_state=42)
                 if hasattr(model, 'random_state')
                 else model.__class__()
                )
            ])
            pipeline.set_params(**params)

            scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=kf, scoring='r2', n_jobs=-1
            )
            return scores.mean()

        # Suppress optuna logs for cleaner output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        print(f"Best CV R² for {model_name}: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")

        self.best_params[model_name] = study.best_params
        return study.best_params

    def _build_tuned_model(self, model_name, best_params):
        """Instantiate model with best hyperparameters from Optuna."""
        if model_name == 'random_forest':
            return RandomForestRegressor(
                random_state=42, n_jobs=-1, **best_params
            )
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                random_state=42, **best_params
            )
        elif model_name == 'ridge':
            return Ridge(**best_params)
        elif model_name == 'lasso':
            return Lasso(**best_params)

    def run_train(self):
        """Full training pipeline."""
        # Load
        df = self.load_data()

        # Prepare
        X, y, numerical_features, categorical_features, date_features = self.prepare_features(df)
        

        # Split — use time-based split since this is temporal data
        # Sort by fyDeclared so future years are always in test set
        df_sorted_idx = df.sort_values('fyDeclared').index
        X = X.loc[df_sorted_idx]
        y = y.loc[df_sorted_idx]

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
        print(f"Train years: {df.loc[X_train.index, 'fyDeclared'].min()} - "
              f"{df.loc[X_train.index, 'fyDeclared'].max()}")
        print(f"Test years:  {df.loc[X_test.index, 'fyDeclared'].min()} - "
              f"{df.loc[X_test.index, 'fyDeclared'].max()}")

        # Train all models
        print("\n" + "="*60)
        print("STEP 1 — Baseline training")
        print("="*60)

        for model_name, model in self.models.items():
            self.train_and_log(
                model_name, model,
                X_train, X_test,
                y_train, y_test,
                numerical_features, 
                categorical_features, 
                date_features,
                tuned=False
            )
        
        print("\n" + "="*60)
        print("STEP 2 — Identifying top models for tuning")
        print("="*60)

        baseline_results = {
            k: v for k, v in self.results.items()
            if not k.endswith('_tuned')
        }
        ranked = sorted(
            baseline_results.items(),
            key=lambda x: x[1]['cv_r2_mean'],
            reverse=True
        )

        top_tunable = [
            name for name, _ in ranked
            if name in TUNABLE_MODELS
        ][:2]

        for model_name in top_tunable:
            base_model = self.models[model_name]

            # Tune on training set only
            best_params = self._tune_model(
                model_name, base_model,
                X_train, y_train,
                numerical_features,
                categorical_features,
                date_features
            )

            # Build tuned model and retrain
            tuned_model = self._build_tuned_model(model_name, best_params)

            self.train_and_log(
                model_name, tuned_model,
                X_train, X_test,
                y_train, y_test,
                numerical_features,
                categorical_features,
                date_features,
                tuned=True
            )

        # Step 4 — final comparison
        best = self.summarize_results()
        return best, self.results[best]['pipeline']




if __name__ == "__main__":
    trainer = ModelTrainer(n_trials=10)
    best_model, best_pipeline = trainer.run_train()