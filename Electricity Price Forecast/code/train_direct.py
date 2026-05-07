"""
Train Direct multi-step forecasting models.

Each target hour owns one independent model and receives its own random
hyperparameter search.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config
from data_split import split_by_months
from feature_engineering_direct import DirectFeatureEngineer
from feature_selector import FeatureSelector
from model_factory import create_model, get_default_params, get_param_space, list_model_types
from utils.metrics import calculate_mae, calculate_rmse, calculate_smape, calculate_accuracy_rate


logger = logging.getLogger(__name__)


class DirectTrainer:
    """Train one independent model per forecast hour."""

    def __init__(
        self,
        config: Config,
        model_type: str,
        n_iter: int,
        cv_folds: int,
        test_months: Optional[List[str]] = None,
        model_dir: Optional[Path] = None,
    ):
        self.config = config
        self.model_type = model_type
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.test_months = test_months or config.split_config.get("test_months")
        self.feature_selector = FeatureSelector()
        self.model_dir = model_dir or (config.get_model_path("direct") / model_type)
        self.feature_importance_dir = config.get_result_path("logs") / "direct" / model_type / "feature_importance"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def prepare_hourly_data(self, hour: int) -> Dict[str, Any]:
        engineer = DirectFeatureEngineer()
        features_df, target_col = engineer.load_features(hour)

        candidate_features = [
            col for col in features_df.columns if col not in [target_col, "预测日期"]
        ]
        numeric_features = features_df[candidate_features].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = self.feature_selector.select_features_for_model(self.model_type, numeric_features, hour=hour)
        feature_info = self.feature_selector.get_model_feature_info(self.model_type)

        split = split_by_months(features_df, "预测日期", self.test_months)

        X_train = features_df.loc[split.train_mask, feature_cols].reset_index(drop=True)
        y_train = features_df.loc[split.train_mask, target_col].to_numpy()
        X_test = features_df.loc[split.test_mask, feature_cols].reset_index(drop=True)
        y_test = features_df.loc[split.test_mask, target_col].to_numpy()
        train_dates = features_df.loc[split.train_mask, "预测日期"].reset_index(drop=True)
        test_dates_series = features_df.loc[split.test_mask, "预测日期"].reset_index(drop=True)

        scaler = None
        if feature_info.get("normalize", False):
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "train_dates": train_dates,
            "test_dates": test_dates_series,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "scaler": scaler,
            "split_info": split.to_dict(),
        }

    def train_hour(self, hour: int) -> Dict[str, Any]:
        logger.info("开始训练 H%02d Direct 模型", hour)
        start = time.time()
        data = self.prepare_hourly_data(hour)

        best_params, best_cv_smape = self._search_best_params(
            data["X_train"],
            data["y_train"],
            hour=hour,
            random_state=42 + hour,
        )

        model = create_model(self.model_type, best_params)
        model.fit(data["X_train"], data["y_train"])

        y_pred = model.predict(data["X_test"])
        test_mae = calculate_mae(data["y_test"], y_pred)
        test_rmse = calculate_rmse(data["y_test"], y_pred)
        test_smape = calculate_smape(data["y_test"], y_pred)
        test_acc_rate = calculate_accuracy_rate(data["y_test"], y_pred, threshold=20.0)

        model_path = self.model_dir / f"model_H{hour:02d}.pkl"
        scaler_path = self.model_dir / f"scaler_H{hour:02d}.pkl"
        metadata_path = self.model_dir / f"metadata_H{hour:02d}.json"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        if data["scaler"] is not None:
            joblib.dump(data["scaler"], scaler_path)
        elif scaler_path.exists():
            scaler_path.unlink()

        feature_importance_path = self._save_feature_importance(hour, model, data["feature_cols"])
        elapsed = time.time() - start
        metadata = {
            "model_type": self.model_type,
            "hour": hour,
            "target_col": data["target_col"],
            "feature_cols": data["feature_cols"],
            "best_params": best_params,
            "split_info": data["split_info"],
            "best_cv_smape": best_cv_smape,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_smape": test_smape,
            "test_acc_rate": test_acc_rate,
            "training_time": elapsed,
            "feature_importance_path": str(feature_importance_path),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "H%02d 完成: CV_sMAPE=%.4f, Test_MAE=%.4f, sMAPE=%.2f%%",
            hour,
            best_cv_smape,
            test_mae,
            test_smape,
        )
        return {
            "hour": hour,
            "status": "success",
            "best_cv_smape": best_cv_smape,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_smape": test_smape,
            "test_acc_rate": test_acc_rate,
            "training_time": elapsed,
            "model_path": str(model_path),
            "feature_importance_path": str(feature_importance_path),
        }

    def train(self, hours: Optional[List[int]] = None) -> pd.DataFrame:
        if hours is None:
            hours = list(range(24))

        results = []
        for hour in hours:
            try:
                results.append(self.train_hour(hour))
            except Exception as exc:
                logger.exception("H%02d 训练失败", hour)
                results.append({"hour": hour, "status": "failed", "error": str(exc)})

        results_df = pd.DataFrame(results)
        self._save_report(results_df)
        self._save_feature_selection_summary(results_df)
        return results_df

    def _save_feature_importance(self, hour: int, model: Any, feature_cols: List[str]) -> Path:
        self.feature_importance_dir.mkdir(parents=True, exist_ok=True)
        raw_importance, source = self._extract_feature_importance(model, feature_cols)

        if not raw_importance:
            raw_importance = {feature: 0.0 for feature in feature_cols}
            source = "unsupported"

        total_importance = float(sum(raw_importance.values()))
        rows = []
        for feature, importance in raw_importance.items():
            normalized = importance / total_importance if total_importance > 0 else 0.0
            rows.append(
                {
                    "hour": hour,
                    "feature": feature,
                    "raw_importance": float(importance),
                    "normalized_importance": float(normalized),
                    "selected": bool(importance != 0),
                    "source": source,
                }
            )

        importance_df = pd.DataFrame(rows)
        importance_df = importance_df.sort_values(
            ["raw_importance", "feature"],
            ascending=[False, True],
        ).reset_index(drop=True)
        path = self.feature_importance_dir / f"feature_importance_H{hour:02d}.csv"
        importance_df.to_csv(path, index=False, encoding="utf-8-sig")

        if source == "unsupported":
            logger.warning("H%02d 模型不支持特征重要性，已按 0 输出: %s", hour, type(model).__name__)
        else:
            logger.info("H%02d 特征重要性已保存: %s", hour, path)
        return path

    def _extract_feature_importance(self, model: Any, feature_cols: List[str]) -> Tuple[Dict[str, float], str]:
        feature_cols = list(feature_cols)

        if hasattr(model, "fitted_members_"):
            aggregate = {feature: 0.0 for feature in self._collect_ensemble_features(model)}
            for weight, member_cols, member_model in model.fitted_members_:
                member_importance, _ = self._extract_feature_importance(member_model, list(member_cols))
                for feature, importance in member_importance.items():
                    aggregate[feature] = aggregate.get(feature, 0.0) + float(weight) * float(importance)
            return aggregate, "feature_ensemble_weighted"

        if hasattr(model, "main_model") and hasattr(model, "low_model"):
            main_importance, _ = self._extract_feature_importance(model.main_model, feature_cols)
            if getattr(model, "has_low_model_", False):
                low_importance, _ = self._extract_feature_importance(model.low_model, feature_cols)
                blend = float(getattr(model, "blend", 0.7))
                main_weight = 1.0 - blend
                low_weight = blend
            else:
                low_importance = {feature: 0.0 for feature in feature_cols}
                main_weight = 1.0
                low_weight = 0.0

            aggregate = {}
            for feature in feature_cols:
                aggregate[feature] = (
                    main_weight * float(main_importance.get(feature, 0.0))
                    + low_weight * float(low_importance.get(feature, 0.0))
                )
            return aggregate, "two_stage_regressor_weighted"

        if hasattr(model, "model"):
            return self._extract_feature_importance(model.model, feature_cols)

        if hasattr(model, "feature_importances_"):
            values = self._align_importance_values(model.feature_importances_, feature_cols, type(model).__name__)
            return dict(zip(feature_cols, values)), "native"

        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            if coef.ndim > 1:
                values = np.sum(np.abs(coef), axis=0)
            else:
                values = np.abs(coef).reshape(-1)
            values = self._align_importance_values(values, feature_cols, type(model).__name__)
            return dict(zip(feature_cols, values)), "linear_coef_abs"

        return {feature: 0.0 for feature in feature_cols}, "unsupported"

    @staticmethod
    def _collect_ensemble_features(model: Any) -> List[str]:
        features = []
        seen = set()
        for _, member_cols, _ in getattr(model, "fitted_members_", []):
            for feature in member_cols:
                if feature not in seen:
                    seen.add(feature)
                    features.append(feature)
        return features

    @staticmethod
    def _align_importance_values(values: Any, feature_cols: List[str], model_name: str) -> List[float]:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if len(arr) != len(feature_cols):
            logger.warning(
                "%s 特征重要性长度与特征数不一致: importance=%s, features=%s",
                model_name,
                len(arr),
                len(feature_cols),
            )
            aligned = np.zeros(len(feature_cols), dtype=float)
            limit = min(len(arr), len(feature_cols))
            if limit:
                aligned[:limit] = arr[:limit]
            arr = aligned
        return [float(value) for value in arr]

    def _save_feature_selection_summary(self, results_df: pd.DataFrame) -> None:
        if "feature_importance_path" not in results_df.columns:
            return

        frames = []
        for row in results_df.itertuples(index=False):
            if getattr(row, "status", None) != "success":
                continue
            path_value = getattr(row, "feature_importance_path", None)
            if not path_value:
                continue
            path = Path(path_value)
            if path.exists():
                frames.append(pd.read_csv(path))

        if not frames:
            logger.warning("没有可汇总的特征重要性文件")
            return

        importance_df = pd.concat(frames, ignore_index=True)
        summary_rows = []
        for feature, group in importance_df.groupby("feature", sort=True):
            included_hours = sorted(int(hour) for hour in group["hour"].unique())
            selected_group = group[group["raw_importance"] != 0]
            selected_hours = sorted(int(hour) for hour in selected_group["hour"].unique())
            unselected_hours = sorted(set(included_hours) - set(selected_hours))
            included_count = len(included_hours)
            selected_count = len(selected_hours)
            unselected_count = len(unselected_hours)
            summary_rows.append(
                {
                    "feature": feature,
                    "included_hours": included_count,
                    "selected_hours": selected_count,
                    "unselected_hours": unselected_count,
                    "selection_rate": selected_count / included_count if included_count else 0.0,
                    "mean_raw_importance": float(group["raw_importance"].mean()),
                    "mean_normalized_importance": float(group["normalized_importance"].mean()),
                    "max_normalized_importance": float(group["normalized_importance"].max()),
                    "hours_included": ",".join(f"H{hour:02d}" for hour in included_hours),
                    "hours_selected": ",".join(f"H{hour:02d}" for hour in selected_hours),
                    "hours_unselected": ",".join(f"H{hour:02d}" for hour in unselected_hours),
                }
            )

        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["selection_rate", "included_hours", "mean_normalized_importance", "feature"],
            ascending=[True, False, False, True],
        )
        path = self.feature_importance_dir / "feature_selection_summary.csv"
        summary_df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info("特征选中率汇总已保存: %s", path)

    def _search_best_params(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        hour: int,
        random_state: int,
    ) -> Tuple[Dict[str, Any], float]:
        default_params = get_default_params(self.model_type, hour=hour)
        param_space = get_param_space(self.model_type)

        if self.n_iter <= 0 or not param_space:
            return default_params, self._cross_val_smape(default_params, X_train, y_train)

        sampler = ParameterSampler(param_space, n_iter=self.n_iter, random_state=random_state)
        best_params = default_params
        best_score = float("inf")

        for i, params in enumerate(sampler, start=1):
            try:
                score = self._cross_val_smape(params, X_train, y_train)
            except Exception as exc:
                logger.warning("参数组合 %s 失败: %s", i, exc)
                continue

            if score < best_score:
                best_score = score
                best_params = dict(params)
                logger.info("  新最优 [%s/%s]: CV_sMAPE=%.4f", i, self.n_iter, score)

        if not np.isfinite(best_score):
            best_score = self._cross_val_smape(best_params, X_train, y_train)

        return best_params, best_score

    def _cross_val_mae(self, params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> float:
        scores = []
        for train_idx, val_idx in self._time_series_folds(len(X), self.cv_folds):
            model = create_model(self.model_type, params)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(calculate_mae(y[val_idx], pred))
        return float(np.mean(scores))

    def _cross_val_smape(self, params: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> float:
        scores = []
        for train_idx, val_idx in self._time_series_folds(len(X), self.cv_folds):
            model = create_model(self.model_type, params)
            model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[val_idx])
            scores.append(calculate_smape(y[val_idx], pred))
        return float(np.mean(scores))

    @staticmethod
    def _time_series_folds(n_samples: int, cv_folds: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        if n_samples < cv_folds + 2:
            raise ValueError("样本量不足，无法进行时间序列交叉验证")

        for fold in range(cv_folds):
            train_end = int(n_samples * (fold + 1) / (cv_folds + 1))
            val_end = int(n_samples * (fold + 2) / (cv_folds + 1))
            if train_end == 0 or val_end <= train_end:
                continue
            yield np.arange(0, train_end), np.arange(train_end, val_end)

    def _save_report(self, results_df: pd.DataFrame) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        report_csv = self.model_dir / "training_report.csv"
        report_json = self.model_dir / "training_report.json"
        results_df.to_csv(report_csv, index=False, encoding="utf-8-sig")
        results_df.to_json(report_json, orient="records", force_ascii=False, indent=2)


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train_direct.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct 多步电价预测训练")
    parser.add_argument("--model", default="lightgbm", choices=list_model_types(), help="基模型类型")
    parser.add_argument("--hours", type=int, nargs="+", default=None, help="指定训练小时，如: --hours 0 8 12")
    parser.add_argument("--n-iter", type=int, default=20, help="每个小时的随机搜索次数；0 表示使用默认参数")
    parser.add_argument("--cv-folds", type=int, default=3, help="时间序列交叉验证折数")
    parser.add_argument("--test-months", nargs="+", default=None, help="测试月份 YYYY-MM；可传多个，默认使用最后一个可用月份")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    trainer = DirectTrainer(Config(), args.model, args.n_iter, args.cv_folds, test_months=args.test_months)
    results = trainer.train(args.hours)

if __name__ == "__main__":
    main()
