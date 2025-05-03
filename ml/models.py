from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

models = {}

models["Linear Regression"] = LinearRegression()

models["Random Forest"] = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

models["Gradient Boosting"] = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=4,
    random_state=42
)

models["SVM"] = SVR(
    kernel='rbf',
    C=10,
    epsilon=0.05
)

models["KNN"] = KNeighborsRegressor(
    n_neighbors=3,
    weights='distance'
)

models["Decision Tree"] = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

try:
    from lightgbm import LGBMRegressor
    models["LightGBM"] = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=42,
        verbose=-1
    )
except ImportError:
    print("ERROR: LightGBM not installed. Skipping LGBMRegressor.")

try:
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
except ImportError:
    print("ERROR: XGBoost not installed. Skipping XGBRegressor.")

def get_models():
    return models
