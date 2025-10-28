# apputil.py
import numpy as np
import pandas as pd


class GroupEstimate:
    """Simple group-level estimator that stores mean/median of y per categorical group."""

    def __init__(self, estimate="mean"):
        if estimate not in ("mean", "median"):
            raise ValueError("estimate must be 'mean' or 'median'")
        self.estimate = estimate
        self._group_map = None
        self._columns = None

    def fit(self, X, y):
        """
        X: pandas DataFrame of categorical columns.
        y: 1D array-like (no missing values).
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        y_ser = pd.Series(y)
        if y_ser.isna().any():
            raise ValueError("y contains missing values")

        if len(X) != len(y_ser):
            raise ValueError("X and y must have the same number of rows")

        # Combine and group
        df = X.copy().reset_index(drop=True)
        df["_y_"] = y_ser.values

        grouped = df.groupby(list(X.columns))["_y_"]

        if self.estimate == "mean":
            agged = grouped.mean()
        else:
            agged = grouped.median()

        # Store mapping from group tuple -> estimate value
        self._group_map = {tuple(idx if isinstance(idx, tuple) else (idx,)): float(val)
                           for idx, val in agged.items()}
        self._columns = list(X.columns)
        return self

    def predict(self, X_):
        """
        X_: array-like or DataFrame of rows with same columns as fit.
        Returns: numpy array of estimates (float), missing -> np.nan
        Also prints a message with number of missing groups.
        """
        if self._group_map is None or self._columns is None:
            raise ValueError("Estimator not fitted. Call .fit(X, y) first.")

        # Convert X_ to DataFrame with correct columns
        if isinstance(X_, pd.DataFrame):
            df_pred = X_.copy().reset_index(drop=True)
        else:
            df_pred = pd.DataFrame(X_, columns=self._columns)

        if list(df_pred.columns) != self._columns:
            # Try to coerce columns if passed as array without names
            if df_pred.shape[1] != len(self._columns):
                raise ValueError("X_ must have the same number of columns as training X")
            df_pred.columns = self._columns

        results = []
        missing_count = 0

        for _, row in df_pred.iterrows():
            key = tuple(row[col] for col in self._columns)
            if key in self._group_map:
                results.append(self._group_map[key])
            else:
                results.append(np.nan)
                missing_count += 1

        if missing_count:
            print(f"Warning: {missing_count} observations had groups not seen in training; returning NaN for those.")

        return np.array(results, dtype=float)
