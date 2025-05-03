from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def get_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=None, random_state=42
        ),
        "Linear Regression": LinearRegression(),
        "SVM": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=None, random_state=42
        )
    }
