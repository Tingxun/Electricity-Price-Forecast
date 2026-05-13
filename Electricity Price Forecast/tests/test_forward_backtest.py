import unittest
import pandas as pd


from EPF.utils.auto_model_selection import AutoCandidate, generate_auto_candidates, monthly_time_series_folds
from EPF.utils.data_split import split_by_months
from EPF.strategies.direct.backtest import DirectMonthlyBacktester
from EPF.config import Config
from EPF.models.factory import create_model, get_default_params, get_param_space, list_model_types
from EPF.strategies.direct.train import DirectTrainer
from EPF.utils.model_store import safe_period_label


class ForwardBacktestTests(unittest.TestCase):
    def setUp(self):
        self.date_col = "预测日期"
        self.df = pd.DataFrame(
            {
                self.date_col: pd.date_range("2024-06-12", "2025-06-30", freq="D"),
                "value": 1.0,
            }
        )

    def test_forward_month_splits_do_not_leak_future_data(self):
        expected_train_end = {
            "2025-03": "2025-02-28",
            "2025-04": "2025-03-31",
            "2025-05": "2025-04-30",
            "2025-06": "2025-05-31",
        }

        for month, train_end in expected_train_end.items():
            with self.subTest(month=month):
                split = split_by_months(self.df, self.date_col, [month])
                self.assertEqual(split.train_end, train_end)
                self.assertLess(pd.Timestamp(split.train_end), pd.Timestamp(split.test_start))

    def test_weekly_backtest_windows_cover_fixed_test_month(self):
        windows = DirectMonthlyBacktester._weekly_windows_for_month(self.df, self.date_col, "2025-03")

        self.assertEqual(
            windows,
            [
                ("W01", "2025-03-01", "2025-03-07"),
                ("W02", "2025-03-08", "2025-03-14"),
                ("W03", "2025-03-15", "2025-03-21"),
                ("W04", "2025-03-22", "2025-03-28"),
                ("W05", "2025-03-29", "2025-03-31"),
            ],
        )

    def test_weekly_backtest_split_uses_only_data_before_window(self):
        train_mask, test_mask, split_info = DirectMonthlyBacktester._split_by_week_window(
            self.df,
            test_month="2025-03",
            week_start="2025-03-08",
            week_end="2025-03-14",
        )

        train_dates = pd.to_datetime(self.df.loc[train_mask, self.date_col])
        test_dates = pd.to_datetime(self.df.loc[test_mask, self.date_col])
        self.assertEqual(split_info["split_strategy"], "weekly_retrain")
        self.assertEqual(split_info["train_end"], "2025-03-07")
        self.assertEqual(split_info["test_start"], "2025-03-08")
        self.assertEqual(split_info["test_end"], "2025-03-14")
        self.assertTrue((train_dates < pd.Timestamp("2025-03-08")).all())
        self.assertTrue(((test_dates >= pd.Timestamp("2025-03-08")) & (test_dates <= pd.Timestamp("2025-03-14"))).all())

    def test_backtester_retrain_frequency_controls_output_prefix(self):
        monthly = DirectMonthlyBacktester(Config(), "lightgbm_auto", 0, 3, 3, retrain_frequency="monthly")
        weekly = DirectMonthlyBacktester(Config(), "lightgbm_auto", 0, 3, 3, retrain_frequency="weekly")

        self.assertEqual(monthly.rolling_mode, "expanding_forward")
        self.assertEqual(monthly.output_prefix, "monthly_backtest")
        self.assertEqual(weekly.rolling_mode, "expanding_forward_weekly")
        self.assertEqual(weekly.output_prefix, "weekly_backtest")
        with self.assertRaises(ValueError):
            DirectMonthlyBacktester(Config(), "lightgbm_auto", 0, 3, 3, retrain_frequency="daily")

    def test_v3_v4_model_aliases_are_not_registered(self):
        self.assertIn("lightgbm_auto", list_model_types())
        self.assertNotIn("lightgbm_smape_probe_v3", list_model_types())
        self.assertNotIn("lightgbm_smape_probe_v4", list_model_types())
        with self.assertRaises(ValueError):
            get_default_params("lightgbm_smape_probe_v3", hour=12)
        with self.assertRaises(ValueError):
            get_default_params("lightgbm_smape_probe_v4", hour=12)

    def test_model_run_period_label_is_filesystem_safe(self):
        self.assertEqual(safe_period_label("2025-03"), "2025-03")
        self.assertEqual(safe_period_label("2025-03,2025-04"), "2025-03__2025-04")

    def test_monthly_time_series_cv_uses_only_past_months(self):
        folds = monthly_time_series_folds(self.df[self.date_col], cv_folds=3)
        months = pd.to_datetime(self.df[self.date_col]).dt.to_period("M")
        for train_idx, val_idx, validation_month in folds:
            with self.subTest(validation_month=validation_month):
                self.assertTrue((months.iloc[train_idx] < pd.Period(validation_month, freq="M")).all())
                self.assertTrue((months.iloc[val_idx] == pd.Period(validation_month, freq="M")).all())

    def test_auto_candidates_gate_midday_and_low_price_structures(self):
        default_params = {"objective": "regression_l1", "n_estimators": 10, "random_state": 42}
        high_prices = [300.0] * len(self.df)
        midday_candidates = generate_auto_candidates(12, default_params, high_prices, cv_folds=None)
        night_candidates = generate_auto_candidates(2, default_params, high_prices, cv_folds=None)

        self.assertTrue(any("direct_midday_regime" in c.feature_groups for c in midday_candidates))
        self.assertFalse(any("direct_midday_regime" in c.feature_groups for c in night_candidates))
        self.assertFalse(any(c.structure == "two_stage_low_price" for c in midday_candidates))

    def test_midday_auto_candidates_include_quantile_and_low_price_variants(self):
        default_params = {"objective": "regression_l1", "n_estimators": 10, "random_state": 42}
        low_prices = [30.0] * len(self.df)
        candidates = generate_auto_candidates(12, default_params, low_prices, cv_folds=None)
        candidate_names = {candidate.name for candidate in candidates}

        self.assertTrue({"quantile_low_weather", "quantile_mid_weather", "quantile_high_weather"} <= candidate_names)
        self.assertIn("single_weather_floor", candidate_names)
        self.assertIn("weighted_weather_strong", candidate_names)
        self.assertIn("weighted_weather_floor", candidate_names)
        self.assertIn("two_stage_low_price", candidate_names)
        self.assertIn("two_stage_low_price_80", candidate_names)
        self.assertIn("two_stage_low_price_100", candidate_names)
        self.assertIn("two_stage_low_price_120_aggressive", candidate_names)
        self.assertIn("quantile_floor_weather", candidate_names)
        self.assertIn("feature_ensemble_quantile_mix", candidate_names)

    def test_h08_auto_candidates_use_non_midday_pool(self):
        default_params = {"objective": "regression_l1", "n_estimators": 10, "random_state": 42}
        low_prices = [30.0] * len(self.df)
        candidates = generate_auto_candidates(8, default_params, low_prices, cv_folds=None)
        candidate_names = {candidate.name for candidate in candidates}

        self.assertNotIn("quantile_mid_weather", candidate_names)
        self.assertIn("h08_low_price_gate", candidate_names)
        self.assertNotIn("h08_quantile_low_base", candidate_names)
        self.assertFalse(any("direct_midday_regime" in candidate.feature_groups for candidate in candidates))

    def test_low_price_gate_caps_predictions_when_gate_fires(self):
        model = create_model(
            "lightgbm_auto",
            {
                "model_kind": "low_price_gate",
                "objective": "regression_l1",
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": 3,
                "num_leaves": 7,
                "min_child_samples": 2,
                "random_state": 42,
                "low_price_threshold": 150.0,
                "gate_prob_threshold": 0.0,
                "gate_prediction_cap": 40.0,
            },
        )
        X = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5]})
        y = pd.Series([20.0, 30.0, 40.0, 300.0, 320.0, 340.0])

        model.fit(X, y)
        pred = model.predict(X)

        self.assertTrue((pred <= 40.0).all())

    def test_lightgbm_auto_search_space_includes_quantile_alpha(self):
        param_space = get_param_space("lightgbm_auto")

        self.assertIn("quantile", param_space["objective"])
        self.assertEqual(param_space["alpha"], [0.05, 0.10, 0.25, 0.35, 0.50, 0.60, 0.75, 0.90])

    def test_quantile_candidate_preserves_objective_but_searches_alpha(self):
        base_params = {
            "objective": "quantile",
            "alpha": 0.35,
            "n_estimators": 120,
            "feature_groups": ["direct_time_midday"],
        }
        tuned_params = {
            "objective": "regression_l1",
            "alpha": 0.90,
            "n_estimators": 300,
            "learning_rate": 0.05,
        }

        merged = DirectTrainer._merge_tuned_params(base_params, tuned_params)

        self.assertEqual(merged["objective"], "quantile")
        self.assertEqual(merged["alpha"], 0.90)
        self.assertEqual(merged["n_estimators"], 300)
        self.assertEqual(merged["learning_rate"], 0.05)

    def test_alpha_search_space_is_hour_aware(self):
        param_space = get_param_space("lightgbm_auto")

        midday_space = DirectTrainer._param_space_for_base(param_space, {"objective": "quantile"}, hour=12)
        h08_space = DirectTrainer._param_space_for_base(param_space, {"objective": "quantile"}, hour=8)
        night_space = DirectTrainer._param_space_for_base(param_space, {"objective": "quantile"}, hour=2)

        self.assertEqual(h08_space["alpha"], [0.05, 0.10, 0.25, 0.35, 0.50])
        self.assertEqual(midday_space["alpha"], [0.50, 0.60, 0.75, 0.90])
        self.assertEqual(night_space["alpha"], [0.05, 0.10, 0.25, 0.35, 0.50])
        self.assertNotIn("objective", h08_space)
        self.assertNotIn("objective", midday_space)
        self.assertNotIn("objective", night_space)

        unlocked_midday_space = DirectTrainer._param_space_for_base(param_space, {"objective": "regression_l1"}, hour=12)
        self.assertEqual(unlocked_midday_space["objective"], ["quantile", "regression_l1"])

    def test_h13_two_stage_candidates_receive_guardrail_penalty(self):
        candidate = AutoCandidate(
            name="two_stage_low_price_80",
            structure="two_stage_low_price",
            feature_groups=[],
            params={"model_kind": "two_stage_low_price"},
        )

        self.assertEqual(DirectTrainer._auto_structure_score_adjustment(candidate, hour=13), 3.0)
        self.assertEqual(DirectTrainer._auto_structure_score_adjustment(candidate, hour=12), 0.0)

    def test_tune_params_false_skips_param_space(self):
        param_space = get_param_space("lightgbm_auto")
        filtered = DirectTrainer._param_space_for_base(
            param_space,
            {"objective": "quantile", "alpha": 0.75, "tune_alpha": False},
            hour=13,
        )

        self.assertNotIn("objective", filtered)
        self.assertNotIn("alpha", filtered)

    def test_h13_candidate_uses_fixed_high_quantile_base_weather(self):
        default_params = {"objective": "regression_l1", "n_estimators": 10, "random_state": 42}
        low_prices = [30.0] * len(self.df)
        candidates = generate_auto_candidates(13, default_params, low_prices, cv_folds=None)
        candidate = next(c for c in candidates if c.name == "h13_quantile_base_weather_high")

        self.assertEqual(candidate.params["objective"], "quantile")
        self.assertEqual(candidate.params["alpha"], 0.75)
        self.assertFalse(candidate.params["tune_alpha"])
        self.assertFalse(candidate.params["tune_params"])
        self.assertEqual(candidate.feature_groups, ["direct_time", "direct_price_lag", "direct_market_window", "direct_weather_window"])

    def test_hour_specific_candidates_for_remaining_midday_bottlenecks(self):
        default_params = {"objective": "regression_l1", "n_estimators": 10, "random_state": 42}
        low_prices = [30.0] * len(self.df)

        h09 = {candidate.name: candidate for candidate in generate_auto_candidates(9, default_params, low_prices, cv_folds=None)}
        h12 = {candidate.name: candidate for candidate in generate_auto_candidates(12, default_params, low_prices, cv_folds=None)}
        h14 = {candidate.name: candidate for candidate in generate_auto_candidates(14, default_params, low_prices, cv_folds=None)}

        self.assertEqual(h09["h09_low_price_gate"].params["model_kind"], "low_price_gate")
        self.assertFalse(h09["h09_low_price_gate"].params["tune_params"])
        self.assertEqual(h12["h12_quantile_base_weather_high"].params["alpha"], 0.75)
        self.assertFalse(h12["h12_quantile_base_weather_high"].params["tune_params"])
        self.assertEqual(h14["h14_weighted_weather_light_fixed"].params["sample_weight_mode"], "light")
        self.assertFalse(h14["h14_weighted_weather_light_fixed"].params["tune_params"])


if __name__ == "__main__":
    unittest.main()
