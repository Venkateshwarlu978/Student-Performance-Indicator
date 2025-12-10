import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logger.info("Splitting training and test input data into X and y")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # -------------------------------------------------------------
            # 🧠 ALL MODELS
            # -------------------------------------------------------------
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # -------------------------------------------------------------
            # 🔥 HYPERPARAMETER GRIDS
            # -------------------------------------------------------------
            params = {
                "Linear Regression": {},
                
                "Lasso": {
                    'alpha': [0.01, 0.1, 1, 5, 10]
                },

                "Ridge": {
                    'alpha': [0.01, 0.1, 1, 5, 10]
                },

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
                    'max_depth': [None, 5, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },

                "Random Forest Regressor": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },

                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.05, 0.1, 0.5, 1],
                    "n_estimators": [50, 100, 200]
                },
            }

            # -------------------------------------------------------------
            # ⚡ MODEL TRAINING + HYPERPARAMETER TUNING
            # -------------------------------------------------------------
            logger.info("Starting hyperparameter tuning on all models...")

            model_report = {}

            for model_name, model in models.items():
                logger.info(f"Tuning hyperparameters for: {model_name}")

                param_grid = params.get(model_name, {})

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1,
                    verbose=0,
                )

                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_

                y_pred = best_model.predict(X_test)
                test_score = r2_score(y_test, y_pred)

                model_report[model_name] = test_score

                logger.info(f"{model_name}: Best Score = {test_score}")

            # -------------------------------------------------------------
            # 🏆 Selecting Best Model
            # -------------------------------------------------------------
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logger.info(f"🏆 Best Model: {best_model_name} with R2 score {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No model achieved acceptable performance (R2 < 0.6).")

            # -------------------------------------------------------------
            # 💾 SAVE MODEL
            # -------------------------------------------------------------
            logger.info("Saving the best model to artifacts/model.pkl")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)

