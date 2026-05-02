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
        calibration = {"scale": 1.0, "bias": 0.0, "clip_min": None}

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

        elapsed = time.time() - start
        metadata = {
            "model_type": self.model_type,
            "hour": hour,
            "target_col": data["target_col"],
            "feature_cols": data["feature_cols"],
            "best_params": best_params,
            "calibration": calibration,
            "split_info": data["split_info"],
            "best_cv_smape": best_cv_smape,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_smape": test_smape,
            "test_acc_rate": test_acc_rate,
            "training_time": elapsed,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(
            "H%02d 完成: CV_sMAPE=%.4f, Test_MAE=%.4f, sMAPE=%.2f%%, AccRate=%.2f%%",
            hour,
            best_cv_smape,
            test_mae,
            test_smape,
            test_acc_rate,
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
        return results_df

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
    def _split_calibration_data(
        X: pd.DataFrame,
        y: np.ndarray,
        calibration_ratio: float = 0.2,
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        split_idx = max(1, int(len(X) * (1 - calibration_ratio)))
        if split_idx >= len(X):
            split_idx = len(X) - 1
        return X.iloc[:split_idx], y[:split_idx], X.iloc[split_idx:], y[split_idx:]

    def _fit_smape_calibration(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """Fit a simple affine calibration on validation predictions."""
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_true = np.asarray(y_true, dtype=float).reshape(-1)

        scale_grid = np.round(np.arange(0.45, 1.26, 0.05), 2)
        bias_grid = np.arange(-80.0, 61.0, 10.0)
        best = {
            "scale": 1.0,
            "bias": 0.0,
            "clip_min": 0.0,
            "validation_smape": calculate_smape(y_true, np.maximum(y_pred, 0.0)),
        }

        for scale in scale_grid:
            for bias in bias_grid:
                calibrated = np.maximum(y_pred * scale + bias, 0.0)
                score = calculate_smape(y_true, calibrated)
                if score < best["validation_smape"]:
                    best = {
                        "scale": float(scale),
                        "bias": float(bias),
                        "clip_min": 0.0,
                        "validation_smape": float(score),
                    }

        return best

    @staticmethod
    def _apply_calibration(y_pred: np.ndarray, calibration: Dict[str, float]) -> np.ndarray:
        calibrated = np.asarray(y_pred, dtype=float) * calibration.get("scale", 1.0)
        calibrated = calibrated + calibration.get("bias", 0.0)
        clip_min = calibration.get("clip_min", None)
        if clip_min is not None:
            calibrated = np.maximum(calibrated, clip_min)
        return calibrated

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
        logger.info("训练报告已保存: %s", report_csv)


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
