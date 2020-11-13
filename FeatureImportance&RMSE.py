import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils import data_string_to_float, status_calc
import numpy as np
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


#Train&Test Split
data_df = pd.read_csv("ketstats_to_train.csv", index_col="calendardate")
data_df.dropna(axis=0, how="any", inplace=True)
features = data_df.columns[1:-4]
X = pd.DataFrame(data_df[features])
y = pd.DataFrame(list(
        status_calc(
            data_df["stock_p_change"], data_df["sp500_p_change"], outperformance=10
        )
    ))

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1
    )

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#Feature Importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns)
feature_importances.sort_values(inplace=True)
ax = feature_importances.plot.barh()
ax.set(ylabel='Importance (Gini Coefficient)', title='Feature importances');
ax.set_title('Feature Importance (MDI)')

#RMSE
print('Root Mean Squared Error:', 
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))