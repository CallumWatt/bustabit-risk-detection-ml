import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve,
    auc
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# -----------------------------
# Data loading / preprocessing
# -----------------------------

def load_data(csv_path: str = "bustabit.csv") -> pd.DataFrame: # load data from csv file
    df = pd.read_csv(csv_path)
    print(df.head())
    print(df.info())
    print(df.describe(include="all"))
    print(df.isnull().sum()) # general database print-out for analysis

    df["PlayDate"] = pd.to_datetime(df["PlayDate"]) # convert to datetime format
    df["Profit"] = df["Profit"].fillna(-df["Bet"]) # fill non-values for calculations
    df["CashedOut"] = df["CashedOut"].fillna(0)
    df["Bonus"] = df["Bonus"].fillna(0)

    return df.sort_values(by=["Username", "PlayDate"]).reset_index(drop=True)


# -----------------------------
# Shared feature engineering
# -----------------------------

def sequence_trend(user_df: pd.DataFrame) -> float:
    """Log-linear slope of bet amounts over bet order."""
    if len(user_df) < 5: # if the user has less than 5 bets, then they will be ignored
        return 0.0

    bet_index = np.arange(len(user_df)).reshape(-1, 1)
    bets = user_df["Bet"].values
    log_bets = np.log(bets + 1)  # avoids having log(0)

    model = LinearRegression()
    model.fit(bet_index, log_bets) #previously it was just the sequence of bets, but this didn't account for relative data so instead a logrithmic way of finding variability is used
    return float(model.coef_[0])


def aggregate_player_stats(source_df: pd.DataFrame) -> pd.DataFrame:
    player_stats = source_df.groupby("Username").agg(
        {
            "Bet": ["mean", "median", "std", "max"],
            "Profit": ["sum", "mean"],
            "GameID": ["count"],
            "PlayDate": ["min", "max"],
        }
    )

    player_stats.columns = [
        "avg_bet",
        "median_bet",
        "bet_variance",
        "max_bet",
        "total_profit",
        "avg_profit",
        "count",
        "session_start",
        "session_end",
    ]

    player_stats = player_stats.reset_index()
    player_stats["bet_variance"] = player_stats["bet_variance"].fillna(0)
    player_stats["bets_per_user"] = player_stats["count"].astype(int)
    player_stats["bet_risk_ratio"] = player_stats["max_bet"] / player_stats["avg_bet"]

    losing_bets_count = source_df[source_df["Profit"] < 0].groupby("Username").size() # Calculate the number of losing bets for each user
    player_stats["losing_bets"] = player_stats["Username"].map(losing_bets_count).fillna(0) # Map these counts to the player_stats DataFrame based on Username
    player_stats["loss_rate"] = player_stats["losing_bets"] / player_stats["count"] # find the loss rates
    player_stats["loss_rate"] = player_stats["loss_rate"].fillna(0) # fill NA that may appear
    player_stats = player_stats.drop(columns=["losing_bets"])

    sequence_trends = source_df.groupby("Username").apply(sequence_trend)
    player_stats["sequence_trend"] = player_stats["Username"].map(sequence_trends)

    losing_players = player_stats[player_stats["total_profit"] < 0]
    if not losing_players.empty:
        loss_threshold_5_percent = losing_players["total_profit"].quantile(0.05)
        player_stats["is_top_5_percent_loss"] = (
            player_stats["total_profit"] <= loss_threshold_5_percent
        ).astype(int)
    else:
        player_stats["is_top_5_percent_loss"] = 0

    player_stats = player_stats.replace([np.inf, -np.inf], 0).fillna(0)
    return player_stats



# Assigning risk labels
def assign_risk_labels(player_stats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    valid_player = player_stats[player_stats["bets_per_user"] >= 4].copy()

    loss_rate_75 = valid_player["loss_rate"].quantile(0.75)
    loss_rate_90 = valid_player["loss_rate"].quantile(0.90)
    bet_risk_75 = valid_player["bet_risk_ratio"].quantile(0.75)
    bet_risk_90 = valid_player["bet_risk_ratio"].quantile(0.90)
    bets_75 = valid_player["bets_per_user"].quantile(0.75)
    bets_90 = valid_player["bets_per_user"].quantile(0.90)
    bets_98 = valid_player["bets_per_user"].quantile(0.98)
    trend_90 = valid_player["sequence_trend"].quantile(0.90)
    loss_threshold_5 = player_stats["total_profit"].quantile(0.05) # this includes all players

    print("=== RISK THRESHOLDS ===")
    print(f"loss_rate_75: {loss_rate_75:.4f}")
    print(f"loss_rate_90: {loss_rate_90:.4f}")
    print(f"bet_risk_75: {bet_risk_75:.4f}")
    print(f"bet_risk_90: {bet_risk_90:.4f}")
    print(f"bets_75: {bets_75:.2f}")
    print(f"bets_90: {bets_90:.2f}")
    print(f"bets_98: {bets_98:.2f}")
    print(f"trend_90: {trend_90:.6f}")
    print(f"loss_threshold_5: {loss_threshold_5:.2f}")

    def calculate_risk_score(row: pd.Series) -> int:
        score = 0

        if row["total_profit"] <= loss_threshold_5:
            score += 3

        if row["bet_risk_ratio"] >= bet_risk_90:
            score += 2
        elif row["bet_risk_ratio"] >= bet_risk_75:
            score += 1

        if row["bets_per_user"] >= bets_90:
            score += 2
        elif row["bets_per_user"] >= bets_75:
            score += 1

        if row["sequence_trend"] >= trend_90:
            score += 2

        return score

    def risk_category(score: int) -> int:
        if score >= 4:
            return 2  # high risk
        if score >= 2:
            return 1  # medium risk
        return 0  # low risk

    labelled = player_stats.copy()
    labelled["risk_score"] = labelled.apply(calculate_risk_score, axis=1)
    labelled["risk"] = labelled["risk_score"].apply(risk_category)

    thresholds = {
        "loss_rate_75": loss_rate_75,
        "loss_rate_90": loss_rate_90,
        "bet_risk_75": bet_risk_75,
        "bet_risk_90": bet_risk_90,
        "bets_75": bets_75,
        "bets_90": bets_90,
        "bets_98": bets_98,
        "trend_90": trend_90,
        "loss_threshold_5": loss_threshold_5,
    }

    return labelled, thresholds


# --------------------------------------
# 70/30 past -> future escalation workflow
# --------------------------------------

def build_future_escalation_dataset(
    df: pd.DataFrame, split_ratio: float = 0.7
) -> pd.DataFrame:
    past_df_list = []
    future_trends = []

    for username, user_df in df.groupby("Username"):
        if len(user_df) < 5:
            continue

        split_idx = int(len(user_df) * split_ratio)
        past = user_df.iloc[:split_idx].copy()
        future = user_df.iloc[split_idx:].copy()

        if len(future) < 2:
            continue

        past_df_list.append(past)
        future_trends.append(
            {
                "Username": username,
                "future_trend": sequence_trend(future),
            }
        )

    if not past_df_list or not future_trends:
        raise ValueError("No valid players found for the 70/30 future escalation split.")

    past_df = pd.concat(past_df_list, ignore_index=True)
    future_labels = pd.DataFrame(future_trends)

    player_stats = aggregate_player_stats(past_df)

    future_threshold_90 = future_labels["future_trend"].quantile(0.90)
    future_labels["future_escalation_flag"] = (
        future_labels["future_trend"] >= future_threshold_90
    ).astype(int)

    merged = player_stats.merge(
        future_labels[["Username", "future_escalation_flag", "future_trend"]],
        on="Username",
        how="inner",
    )

    print("=== FUTURE ESCALATION THRESHOLD ===")
    print(f"future_threshold_90: {future_threshold_90:.6f}")
    print(merged[["Username", "future_escalation_flag", "future_trend"]].head())

    return merged


# -----------------------------
# Modelling helpers
# -----------------------------

def evaluate_model(name: str, y_true, y_pred, cv_f1=None, pr_auc=None) -> dict:
    average_type = "binary" if len(np.unique(y_true)) == 2 else "weighted"
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if average_type == "binary":
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "F1 (macro)": macro_f1,
        "CV F1": cv_f1,
        "PR-AUC": pr_auc,
    }
def run_shap_analysis(model, X_train, X_test, feature_names):
    print("\n===== GENERATING SHAP EXPLANATIONS =====")

    # 1. Use the TreeExplainer
    explainer = shap.TreeExplainer(model)

    # 2. Calculate values
    shap_values = explainer.shap_values(X_test)

    # 3. CRITICAL: Handle multi-class indexing
    # For RandomForest (Risk 0, 1, 2), shap_values is a list of 3 matrices.
    # We want index 2 (High Risk).
    if isinstance(shap_values, list):
        display_values = shap_values[2]
    else:
        # For some newer versions of SHAP, it returns a single 3D array
        # (observations, features, classes). We slice for class 2. (assuming it's a binary case or a specific class explanation)
        display_values = shap_values[:, :, 2] if len(shap_values.shape) == 3 else shap_values

    # 4. Force to NumPy to avoid 'Legacy' error
    # SHAP sometimes errors if it receives a pandas DataFrame with metadata
    test_array = X_test.values if hasattr(X_test, "values") else X_test

    plt.figure(figsize=(12, 8))

    # Using 'beeswarm' specifically instead of 'summary_plot'
    # often fixes the keyword error
    shap.summary_plot(
        display_values,
        test_array,
        feature_names=feature_names,
        plot_type="dot",   # Should now be accepted
        alpha=0.5,    # Transparency
        show=False
    )

    plt.title("Drivers of High-Risk Gambling Behavior")
    plt.tight_layout()
    plt.show()

    return explainer, shap_values

def run_shap_analysis_for_non_tree_model(model, X_train_scaled, X_test_scaled, feature_names, model_name="Model", positive_class_idx=2):
    print(f"\n===== GENERATING SHAP EXPLANATIONS FOR {model_name} =====")

    # Define a prediction function for the specific class
    # KernelExplainer expects a function that outputs a single value for each sample
    # or a 2D array of (n_samples, n_outputs) if multiple outputs are explained at once.
    # For explaining a single class in a multi-class problem, we extract its probability.
    def predict_proba_for_class(X):
        return model.predict_proba(X)[:, positive_class_idx]

    # Use KernelExplainer for non-tree models
    # It requires a background dataset for integration
    # Use a sample of the training data as the background to speed up calculation
    explainer = shap.KernelExplainer(predict_proba_for_class, shap.sample(X_train_scaled, 100))

    # Calculate SHAP values for the test set
    # Use a sample of the test data for faster plotting
    shap_values = explainer.shap_values(shap.sample(X_test_scaled, 100))

    # For single-output explainers (which we've made it by wrapping predict_proba_for_class),
    # shap_values is a single array, not a list.
    display_values = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        display_values,
        shap.sample(X_test_scaled, 100), # Plot with the same sample as shap_values
        feature_names=feature_names,
        plot_type="dot",
        alpha=0.5,
        show=False
    )
    plt.title(f"Drivers of High-Risk Gambling Behavior ({model_name})")
    plt.tight_layout()
    plt.show()

    return explainer, shap_values


def evaluate_cv(model, X, y, name: str, scaled: bool = False) -> float:
    estimator = clone(model)
    if scaled:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    scoring = "f1" if len(np.unique(y)) == 2 else "f1_weighted"
    scores = cross_val_score(estimator, X, y, cv=5, scoring=scoring, n_jobs=-1)
    print(f"{name} mean CV {scoring}: {scores.mean():.4f}")
    return float(scores.mean())



def error_analysis(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    focus_class: int | None = None,
) -> pd.DataFrame:
    analysis_df = X_test.copy()
    analysis_df["true_label"] = y_test.values
    analysis_df["predicted_label"] = y_pred

    print(f"\n===== ERROR ANALYSIS: {model_name} =====")
    errors = analysis_df[analysis_df["true_label"] != analysis_df["predicted_label"]]
    print(f"Total incorrect predictions: {len(errors)}")
    print(errors.head())

    if focus_class is not None:
        fn = analysis_df[
            (analysis_df["true_label"] == focus_class)
            & (analysis_df["predicted_label"] != focus_class)
        ]
        fp = analysis_df[
            (analysis_df["true_label"] != focus_class)
            & (analysis_df["predicted_label"] == focus_class)
        ]

        print(f"\nFalse Negatives for class {focus_class}: {len(fn)}")
        print(fn.head())
        print(f"\nFalse Positives for class {focus_class}: {len(fp)}")
        print(fp.head())

        if not fn.empty:
            print("\nFalse Negatives - Mean values:")
            print(fn.mean(numeric_only=True))
        if not fp.empty:
            print("\nFalse Positives - Mean values:")
            print(fp.mean(numeric_only=True))

    return analysis_df



def train_and_compare_models(
    dataset_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    focus_class: int | None = None,
) -> dict:
    if X.empty or y.empty:
        raise ValueError(f"{dataset_name}: X or y is empty.")

    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10, stratify=y
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    is_binary = len(np.unique(y)) == 2

    lr = LogisticRegression(max_iter=1_000_000, class_weight="balanced")
    rf = RandomForestClassifier(
        max_depth=10 if not is_binary else None,
        min_samples_leaf=2,
        min_samples_split=10,
        n_estimators=50 if not is_binary else 100,
        class_weight="balanced" if not is_binary else "balanced", # Balanced for both binary and multiclass
        random_state=42,
    )
    svc = SVC(class_weight="balanced", probability=True, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)

    # Train
    lr.fit(x_train_scaled, y_train)
    rf.fit(x_train_scaled, y_train)
    svc.fit(x_train_scaled, y_train)
    knn.fit(x_train_scaled, y_train)

    # Predict
    lr_pred = lr.predict(x_test_scaled)
    rf_pred = rf.predict(x_test_scaled)
    svc_pred = svc.predict(x_test_scaled)
    knn_pred = knn.predict(x_test_scaled)

    lr_prauc = None
    rf_prauc = None
    svc_prauc = None
    knn_prauc = None

    if is_binary:
      lr_probs = lr.predict_proba(x_test_scaled)[:, 1]
      rf_probs = rf.predict_proba(x_test_scaled)[:, 1]
      svc_probs = svc.predict_proba(x_test_scaled)[:, 1]
      knn_probs = knn.predict_proba(x_test_scaled)[:, 1]

      lr_prauc = average_precision_score(y_test, lr_probs)
      rf_prauc = average_precision_score(y_test, rf_probs)
      svc_prauc = average_precision_score(y_test, svc_probs)
      knn_prauc = average_precision_score(y_test, knn_probs)

      print("\n===== PR-AUC SCORES =====")
      print(f"LogReg:       {lr_prauc:.4f}")
      print(f"RandomForest:{rf_prauc:.4f}")
      print(f"SVC:         {svc_prauc:.4f}")
      print(f"KNN:         {knn_prauc:.4f}")

    print(f"\n===== {dataset_name}: LOGISTIC REGRESSION =====")
    print(classification_report(y_test, lr_pred, zero_division=0))
    print(confusion_matrix(y_test, lr_pred))

    print(f"\n===== {dataset_name}: RANDOM FOREST =====")
    print(classification_report(y_test, rf_pred, zero_division=0))
    print(confusion_matrix(y_test, rf_pred))

    print(f"\n===== {dataset_name}: SVC =====")
    print(classification_report(y_test, svc_pred, zero_division=0))
    print(confusion_matrix(y_test, svc_pred))

    print(f"\n===== {dataset_name}: KNN =====")
    print(classification_report(y_test, knn_pred, zero_division=0))
    print(confusion_matrix(y_test, knn_pred))

    # CV scores
    rf_score = evaluate_cv(rf, X, y, f"{dataset_name} Random Forest", scaled=True)
    knn_score = evaluate_cv(knn, X, y, f"{dataset_name} KNN", scaled=True)
    svc_score = evaluate_cv(svc, X, y, f"{dataset_name} SVC", scaled=True)
    lr_score = evaluate_cv(lr, X, y, f"{dataset_name} LogReg", scaled=True)

    results = pd.DataFrame(
    [
        evaluate_model("LogReg", y_test, lr_pred, lr_score, lr_prauc),
        evaluate_model("RandomForest", y_test, rf_pred, rf_score, rf_prauc),
        evaluate_model("SVC", y_test, svc_pred, svc_score, svc_prauc),
        evaluate_model("KNN", y_test, knn_pred, knn_score, knn_prauc),
    ]
).sort_values(by="F1", ascending=False)

    print(f"\n===== {dataset_name}: MODEL COMPARISON =====")
    print(results)

    # Probability demo for one instance
    if len(x_test) > 5:
        single_instance_scaled = x_test_scaled[5]
        probs = lr.predict_proba([single_instance_scaled])[0]
        class_map = dict(enumerate(lr.classes_))
        print(f"\n===== {dataset_name}: SINGLE INSTANCE PROBABILITIES =====")
        for idx, prob in enumerate(probs):
            print(f"Class {class_map[idx]}: {prob * 100:.2f}%")
        print("Features for selected instance:")
        print(x_test.iloc[5])
        print(f"True label: {y_test.iloc[5]}")

    # Error analysis
    error_analysis(x_test, y_test, rf_pred, "Random Forest", focus_class=focus_class)
    error_analysis(x_test, y_test, knn_pred, "KNN", focus_class=focus_class)
    error_analysis(x_test, y_test, svc_pred, "SVC", focus_class=focus_class)
    error_analysis(x_test, y_test, lr_pred, "LogReg", focus_class=focus_class)

    return {
        "results": results,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "x_train_scaled": x_train_scaled,
        "x_test_scaled": x_test_scaled,
        "scaler": scaler,
        "lr": lr,
        "rf": rf,
        "svc": svc,
        "knn": knn,
        "lr_pred": lr_pred,
        "rf_pred": rf_pred,
        "svc_pred": svc_pred,
        "knn_pred": knn_pred,
    }


# -----------------------------
# Visuals / clustering
# -----------------------------
def plot_precision_recall_curves(models, X_test, y_test, positive_class: int = 2):
    """
    Plots PR curves for multiple models.
    `positive_class` specifies which class to consider as positive for PR curve calculation.
    """
    plt.figure(figsize=(10, 7))

    for name, model in models.items():
        # Get probabilities for the specified positive class
        if hasattr(model, "predict_proba"):
            if model.predict_proba(X_test).shape[1] > 1:
                # For multi-class or binary where positive_class is > 0
                probs = model.predict_proba(X_test)[:, positive_class]
            else:
                # For binary classification where positive_class is 1
                probs = model.predict_proba(X_test)[:, 1]

            # Convert y_test to binary (1 if positive_class, else 0) for this specific plot
            y_binary = (y_test == positive_class).astype(int)

            precision, recall, _ = precision_recall_curve(y_binary, probs)
            pr_auc = auc(recall, precision)

            plt.plot(recall, precision, label=f'{name} (PR-AUC = {pr_auc:.2f})')

    plt.xlabel('Recall (Ability to find positive class players)')
    plt.ylabel('Precision (Accuracy of positive class flags)')
    plt.title(f'Precision-Recall Curve: Class {positive_class}')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()

def plot_risk_distributions(player_stats: pd.DataFrame) -> None:
    valid_player = player_stats[player_stats["bets_per_user"] >= 4]

    sns.countplot(x="risk", data=player_stats)
    plt.title("Risk Category Distribution")
    plt.show()

    sns.histplot(valid_player["loss_rate"], bins=50)
    plt.title("Loss Rates")
    plt.show()

    sns.histplot(valid_player["bets_per_user"], bins=50)
    plt.title("Bets Per User")
    plt.show()

    sns.boxplot(x="risk", y="loss_rate", data=player_stats)
    plt.title("Loss Rate by Risk Category")
    plt.show()

    sns.boxplot(x="risk", y="bet_risk_ratio", data=player_stats)
    plt.title("Bet Risk Ratio by Risk Category")
    plt.show()

    sns.boxplot(x="risk", y="bets_per_user", data=player_stats)
    plt.title("Bets Per User by Risk Category")
    plt.show()

    corr = player_stats[[
        "avg_bet",
        "median_bet",
        "bet_variance",
        "max_bet",
        "total_profit",
        "bets_per_user",
        "loss_rate",
        "bet_risk_ratio",
        "sequence_trend",
    ]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Matrix")
    plt.show()

    sns.histplot(player_stats["total_profit"], bins=50)
    plt.title("Profit per Player")
    plt.show()

    sns.histplot(player_stats["sequence_trend"], bins=50)
    plt.title("Bet Escalation Trend Distribution")
    plt.show()



def run_kmeans_analysis(
    player_stats: pd.DataFrame,
    feature_columns: list[str],
    rf_model: RandomForestClassifier,
) -> None:
    X = player_stats[feature_columns].replace([np.inf, -np.inf], 0).fillna(0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(x_scaled)
        inertia.append(kmeans.inertia_)

    plt.plot(list(k_range), inertia, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.show()

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    player_stats["cluster"] = kmeans.fit_predict(x_scaled)

    print(pd.crosstab(player_stats["cluster"], player_stats["risk"]))

    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x_scaled)

    plt.figure()
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], hue=player_stats["cluster"], palette="Set1")
    plt.title("KMeans Clusters (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    print(player_stats.groupby("cluster").mean(numeric_only=True))

    player_stats["rf_pred"] = rf_model.predict(X)
    print(pd.crosstab(player_stats["cluster"], player_stats["rf_pred"]))
    print(pd.crosstab(player_stats["risk"], player_stats["rf_pred"]))

    ct = pd.crosstab(player_stats["cluster"], player_stats["rf_pred"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="Greens")
    plt.title("Cluster vs Random Forest Predictions")
    plt.show()

    importance = pd.Series(rf_model.feature_importances_, index=feature_columns).sort_values()
    importance.plot(kind="barh")
    plt.title("Feature Importance (Random Forest)")
    plt.show()


# -----------------------------
# Main
# -----------------------------

def main():
    df = load_data("bustabit.csv")

    feature_columns = [
        "avg_bet",
        "median_bet",
        "bet_variance",
        "max_bet",
        "total_profit",
        "bets_per_user",
        "loss_rate",
        "bet_risk_ratio",
        "sequence_trend",
    ]

    # Workflow 1: original multiclass risk classification
    full_player_stats = aggregate_player_stats(df)
    full_player_stats, _ = assign_risk_labels(full_player_stats)
    plot_risk_distributions(full_player_stats)

    risk_model_outputs = train_and_compare_models(
        dataset_name="Risk classification",
        X=full_player_stats[feature_columns],
        y=full_player_stats["risk"],
        focus_class=2,
    )

    trained_models_risk = {
        "Random Forest": risk_model_outputs["rf"],
        "Logistic Regression": risk_model_outputs["lr"],
        "SVC": risk_model_outputs["svc"],
        "KNN": risk_model_outputs["knn"]
    }
    plot_precision_recall_curves(trained_models_risk, risk_model_outputs["x_test_scaled"], risk_model_outputs["y_test"], positive_class=2)

    run_kmeans_analysis(
        player_stats=full_player_stats,
        feature_columns=feature_columns,
        rf_model=risk_model_outputs["rf"],
    )

    run_shap_analysis(
      model=risk_model_outputs["rf"],
      X_train=risk_model_outputs["x_train"],
      X_test=risk_model_outputs["x_test"],
      feature_names=feature_columns
    )

    # Add SHAP analysis for SVC
   

    # Workflow 2: 70/30 past -> future escalation prediction
    future_player_stats = build_future_escalation_dataset(df, split_ratio=0.7)
    future_model_outputs = train_and_compare_models(
        dataset_name="Future escalation (70/30 split)",
        X=future_player_stats[feature_columns],
        y=future_player_stats["future_escalation_flag"],
        focus_class=1,
    )

    run_shap_analysis_for_non_tree_model(
      model=future_model_outputs["svc"],
      X_train_scaled=future_model_outputs["x_train_scaled"],
      X_test_scaled=future_model_outputs["x_test_scaled"],
      feature_names=feature_columns,
      model_name="SVC Future Escalation",
      positive_class_idx=1
    )

    trained_models_future = {
        "Random Forest": future_model_outputs["rf"],
        "Logistic Regression": future_model_outputs["lr"],
        "SVC": future_model_outputs["svc"],
        "KNN": future_model_outputs["knn"]
    }
    plot_precision_recall_curves(trained_models_future, future_model_outputs["x_test_scaled"], future_model_outputs["y_test"], positive_class=1)

    print("\nMerged script finished successfully.")
    print("Best models table for risk classification:")
    print(risk_model_outputs["results"])
    print("\nBest models table for future escalation:")
    print(future_model_outputs["results"])


if __name__ == "__main__":
    main()