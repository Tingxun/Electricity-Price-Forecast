"""Train true MIMO models that predict the full 24-hour price curve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from config import Config
from feature_engineering_mimo import MimoFeatureEngineer
from model_tcn_mimo import TcnMimoConfig, TcnMimoNet, resolve_device
from utils.evaluation import prediction_rows_from_wide, summarize_predictions
from utils.model_store import safe_period_label


logger = logging.getLogger(__name__)

SUPPORTED_MIMO_MODELS = ["tcn_mimo"]


class MimoTrainer:
    def __init__(
        self,
        config: Config,
        model_type: str = "tcn_mimo",
        test_months: Optional[List[str]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Path] = None,
    ):
        if model_type not in SUPPORTED_MIMO_MODELS:
            raise ValueError(f"unsupported MIMO model_type: {model_type}")
        self.config = config
        self.model_type = model_type
        self.test_months = test_months or config.split_config.get("test_months")
        self.model_config = TcnMimoConfig.from_dict(model_config or {})
        self.model_dir = model_dir

    def train(self) -> Dict[str, Any]:
        start = time.time()
        samples = MimoFeatureEngineer(lookback_days=self.model_config.lookback_days).load_features()
        split = self._month_split(samples["dates"], self.test_months)
        train_idx, val_idx = self._train_val_split(samples["dates"], split["train_idx"])
        self._ensure_model_dir(split["test_period"])

        scalers = self._fit_scalers(samples, train_idx)
        tensors = self._build_tensors(samples, scalers)
        device = resolve_device(self.model_config.device)
        torch.manual_seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)

        model = TcnMimoNet(samples["target_exog"].shape[2], self.model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.model_config.lr,
            weight_decay=self.model_config.weight_decay,
        )
        train_loader = DataLoader(
            TensorDataset(
                tensors["price"][train_idx],
                tensors["exog"][train_idx],
                tensors["y_scaled"][train_idx],
            ),
            batch_size=self.model_config.batch_size,
            shuffle=True,
        )

        best_state = None
        best_score = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, self.model_config.epochs + 1):
            train_loss = self._train_one_epoch(model, train_loader, optimizer, device, scalers)
            val_pred = self._predict_scaled(model, tensors, val_idx, device)
            val_pred_orig = self._inverse_target(val_pred, scalers)
            val_pred_orig = self._clip_predictions(val_pred_orig, scalers)
            val_actual = samples["targets"][val_idx]
            val_score = self._robust_score(samples["dates"][val_idx], val_actual, val_pred_orig)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_robust_score": val_score})

            if val_score < best_score:
                best_score = val_score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.model_config.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        test_pred = self._predict_scaled(model, tensors, split["test_idx"], device)
        test_pred_orig = self._clip_predictions(self._inverse_target(test_pred, scalers), scalers)
        test_actual = samples["targets"][split["test_idx"]]
        test_dates = samples["dates"][split["test_idx"]]
        prediction_df = prediction_rows_from_wide(test_dates, test_actual, test_pred_orig)
        summary = summarize_predictions(prediction_df)

        model_path = self.model_dir / "model.pt"
        torch.save({"state_dict": model.state_dict(), "model_config": self.model_config.__dict__}, model_path)
        metadata = {
            "strategy": "mimo",
            "model_type": self.model_type,
            "test_period": split["test_period"],
            "test_months": split["test_months"],
            "feature_path": str(self.config.get_data_path("features") / "mimo" / "mimo_samples.npz"),
            "model_config": self.model_config.__dict__,
            "exog_columns": samples["exog_columns"],
            "scalers": scalers,
            "split_info": {
                "train_start": str(samples["dates"][split["train_idx"]][0]),
                "train_end": str(samples["dates"][split["train_idx"]][-1]),
                "validation_start": str(samples["dates"][val_idx][0]),
                "validation_end": str(samples["dates"][val_idx][-1]),
                "test_start": str(samples["dates"][split["test_idx"]][0]),
                "test_end": str(samples["dates"][split["test_idx"]][-1]),
                "n_train_window": int(len(split["train_idx"])),
                "n_fit": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                "n_test": int(len(split["test_idx"])),
            },
            "best_epoch": int(best_epoch),
            "best_validation_robust_score": float(best_score),
            "test_metrics": summary["overall"],
            "training_history": history,
            "training_time": time.time() - start,
            "model_path": str(model_path),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(self.model_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        report = pd.DataFrame(
            [
                {
                    "model_type": self.model_type,
                    "test_period": split["test_period"],
                    "best_epoch": best_epoch,
                    "best_validation_robust_score": best_score,
                    **summary["overall"],
                    "training_time": time.time() - start,
                    "model_path": str(model_path),
                }
            ]
        )
        report.to_csv(self.model_dir / "training_report.csv", index=False, encoding="utf-8-sig")
        logger.info("MIMO training completed: %s", report.to_dict("records")[0])
        return metadata

    def _ensure_model_dir(self, test_period: str) -> None:
        if self.model_dir is None:
            self.model_dir = self.config.project_root / "saved_models" / "mimo" / self.model_type / safe_period_label(test_period)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _month_split(dates: Sequence[str], test_months: Optional[Sequence[str]]) -> Dict[str, Any]:
        date_series = pd.to_datetime(pd.Series(dates))
        months = date_series.dt.to_period("M")
        unique_months = sorted(months.unique())
        if test_months is None:
            selected = [unique_months[-1]]
        else:
            selected = sorted({pd.Period(month, freq="M") for month in test_months})
        first_test = min(selected)
        train_mask = date_series < first_test.start_time
        test_mask = months.isin(selected)
        if not train_mask.any() or not test_mask.any():
            raise ValueError("invalid MIMO month split")
        return {
            "train_idx": np.where(train_mask.to_numpy())[0],
            "test_idx": np.where(test_mask.to_numpy())[0],
            "test_months": [str(month) for month in selected],
            "test_period": ",".join(str(month) for month in selected),
        }

    @staticmethod
    def _train_val_split(dates: Sequence[str], train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        train_dates = pd.to_datetime(pd.Series(dates[train_idx]))
        train_months = train_dates.dt.to_period("M")
        unique_months = sorted(train_months.unique())
        if len(unique_months) >= 2:
            val_month = unique_months[-1]
            val_mask = train_months == val_month
            fit_mask = ~val_mask
            return train_idx[fit_mask.to_numpy()], train_idx[val_mask.to_numpy()]

        split_at = max(1, int(len(train_idx) * 0.8))
        return train_idx[:split_at], train_idx[split_at:]

    def _fit_scalers(self, samples: Dict[str, Any], fit_idx: np.ndarray) -> Dict[str, Any]:
        price_values = samples["price_history"][fit_idx].reshape(-1)
        y_values = samples["targets"][fit_idx].reshape(-1)
        exog_values = samples["target_exog"][fit_idx].reshape(-1, samples["target_exog"].shape[2])
        target_mean = float(y_values.mean())
        target_std = float(y_values.std() + 1e-6)
        exog_mean = exog_values.mean(axis=0).astype(float)
        exog_std = (exog_values.std(axis=0) + 1e-6).astype(float)
        return {
            "price_mean": float(price_values.mean()),
            "price_std": float(price_values.std() + 1e-6),
            "target_mean": target_mean,
            "target_std": target_std,
            "exog_mean": exog_mean.tolist(),
            "exog_std": exog_std.tolist(),
            "clip_min": 0.0,
            "clip_max": float(np.quantile(y_values, 0.999) * 1.2),
        }

    @staticmethod
    def _build_tensors(samples: Dict[str, Any], scalers: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        price = (samples["price_history"] - scalers["price_mean"]) / scalers["price_std"]
        exog = (samples["target_exog"] - np.asarray(scalers["exog_mean"])) / np.asarray(scalers["exog_std"])
        y_scaled = (samples["targets"] - scalers["target_mean"]) / scalers["target_std"]
        return {
            "price": torch.tensor(price, dtype=torch.float32),
            "exog": torch.tensor(exog, dtype=torch.float32),
            "y_scaled": torch.tensor(y_scaled, dtype=torch.float32),
        }

    def _train_one_epoch(self, model, loader, optimizer, device, scalers: Dict[str, Any]) -> float:
        model.train()
        losses = []
        target_mean = torch.tensor(scalers["target_mean"], dtype=torch.float32, device=device)
        target_std = torch.tensor(scalers["target_std"], dtype=torch.float32, device=device)
        for price, exog, y_scaled in loader:
            price = price.to(device)
            exog = exog.to(device)
            y_scaled = y_scaled.to(device)
            optimizer.zero_grad()
            pred_scaled = model(price, exog)
            huber = torch.nn.functional.smooth_l1_loss(pred_scaled, y_scaled)
            pred_orig = pred_scaled * target_std + target_mean
            y_orig = y_scaled * target_std + target_mean
            smape_proxy = (torch.abs(pred_orig - y_orig) / ((torch.abs(pred_orig) + torch.abs(y_orig)) / 2.0 + 1.0)).mean()
            shape = torch.nn.functional.smooth_l1_loss(torch.diff(pred_scaled, dim=1), torch.diff(y_scaled, dim=1))
            loss = (
                self.model_config.huber_weight * huber
                + self.model_config.smape_weight * smape_proxy
                + self.model_config.shape_weight * shape
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses)) if losses else float("inf")

    @staticmethod
    def _predict_scaled(model, tensors: Dict[str, torch.Tensor], indices: np.ndarray, device: torch.device) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(indices), 256):
                batch_idx = indices[start : start + 256]
                pred = model(tensors["price"][batch_idx].to(device), tensors["exog"][batch_idx].to(device))
                preds.append(pred.detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

    @staticmethod
    def _inverse_target(y_scaled: np.ndarray, scalers: Dict[str, Any]) -> np.ndarray:
        return y_scaled * scalers["target_std"] + scalers["target_mean"]

    @staticmethod
    def _clip_predictions(pred: np.ndarray, scalers: Dict[str, Any]) -> np.ndarray:
        return np.clip(pred, scalers["clip_min"], scalers["clip_max"])

    @staticmethod
    def _robust_score(dates: Sequence[str], actual: np.ndarray, pred: np.ndarray) -> float:
        prediction_df = prediction_rows_from_wide(dates, actual, pred)
        summary = summarize_predictions(prediction_df)
        return float(summary["robust_score"])


def setup_logging() -> None:
    log_dir = Config().get_result_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "train_mimo.log", encoding="utf-8"), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MIMO models")
    parser.add_argument("--model", default="tcn_mimo", choices=SUPPORTED_MIMO_MODELS)
    parser.add_argument("--test-months", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    overrides = {
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    MimoTrainer(Config(), args.model, args.test_months, {k: v for k, v in overrides.items() if v is not None}).train()


if __name__ == "__main__":
    main()
