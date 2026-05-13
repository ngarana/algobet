docs/FEATURES_OPTIMIZATIONs.md documents the current state of development, I have completed the deep refactor of both the training pipeline and the backtest runner. The 100% Home-bias
  collapse you saw in the previous backtest was a "false negative" caused by feature scaling drift—the model was being fed incorrectly scaled data during the backtest because the original
  pipeline state was not being preserved.

  Key Fixes Implemented:
  Pipeline Persistence: The FeaturePipeline (scalers, encoders) is now saved as a permanent artifact alongside the model.
  Backtest Hardening: The BacktestRunner now strictly requires the original fitted pipeline. It will no longer "silently fail" by re-fitting on a tiny 30-match slice of data.
  Tuner Integrity: Fixed a critical bug in tuner.py where Optuna was ignoring draw-recall penalties for 90% of the training history. It now optimizes for draw detection across all cross-
  validation windows.
  Draw Signal Amplification: Increased the Dixon-Coles Poisson blend weight to 0.8, allowing the physics-based model to override the classifier's Home-bias on low-entropy matchups. and t
  he following are the results ..curl -sS -X POST http://localhost:8010/api/v1/ml/train \
  ─╯
    -H "Content-Type: application/json" \
    -d '{
    "model_type": "xgboost",
    "description": "V3.5 Final Draw-Recall Optimization",
    "activate": true,
    "tournament_ids": [359],
    "min_matches": 1000,
    "feature_groups": [
      "team_form",
      "enriched_stats",
      "draw_signals",
      "matchup_interaction",
      "temporal"
    ],
    "split_strategy": "season_aware",
    "train_seasons": 4,
    "val_seasons": 1,
    "test_seasons": 1,
    "gap_days": 14,
    "outcome_balance": true,
    "outcome_balance_strength": 0.8,
    "calibrate_probabilities": true,
    "use_cv_calibration": true,
    "calibration_method": "sigmoid",
    "fit_draw_aware_calibrator": true,
    "tune_hyperparameters": true,
    "tuning_trials": 50
  }'

  {"model_id":188,"model_version":"xgboost_20260512_092536","model_type":"xgboost","is_active":false,"feature_schema_version":"v3.1_standings_enriched_fix","num_features":107,"trained_at"
  :"2026-05-12T09:25:36.115765","training_duration_seconds":334.2261254787445,"train_metrics":
  {"accuracy":0.5927631578947369,"log_loss":1.0213568210601807,"precision_macro":0.5943405289788268,"recall_macro":0.5516709201762989,"f1_macro":0.5599883986983144,"predicted_classes":3.0
  ,"max_prediction_share":0.5756578947368421,"expected_calibration_error":0.2849173302909261,"maximum_calibration_error":0.40545840999659366,"brier_score":0.20484802179715098,"market_samp
  les":1520.0,"market_log_loss":0.949915719763708,"market_favorite_accuracy":0.5644736842105263,"market_model_probability_mae":0.13201852295868888,"market_favorite_agreement":0.5223684210
  526316},"val_metrics":
  {"accuracy":0.38421052631578945,"log_loss":1.089850902557373,"precision_macro":0.3642040802960343,"recall_macro":0.35263929618768325,"f1_macro":0.3291965934115791,"predicted_classes":3.
  0,"max_prediction_share":0.6868421052631579,"expected_calibration_error":0.16205351682085745,"maximum_calibration_error":0.5885466933250427,"brier_score":0.22032217132596169,"market_sam
  ples":380.0,"market_log_loss":0.967086621706789,"market_favorite_accuracy":0.5526315789473685,"market_model_probability_mae":0.13328265354904265,"market_favorite_agreement":0.5210526315
  789473},"test_metrics":
  {"accuracy":0.4169014084507042,"log_loss":1.0871731042861938,"precision_macro":0.34185829930510786,"recall_macro":0.34852114209589197,"f1_macro":0.29553369391706574,"predicted_classes":
  3.0,"max_prediction_share":0.7943661971830986,"expected_calibration_error":0.13048402709020696,"maximum_calibration_error":0.29757705330848694,"brier_score":0.21962888968409647,"market_
  goal_variance_10":1.9032738208770752,"home_points_volatility_10":2.2227232456207275,"away_win_rate_10":2.2480132579803467,"away_goals_for_avg_10":2.480760097503662,"away_clean_sheet_rat
  e_10":2.2593557834625244,"away_goal_variance_10":2.284137010574341,"away_points_volatility_10":2.0386533737182617,"home_home_form_5":2.0291554927825928,"away_away_form_5":1.646299958229
  065,"home_home_form_10":1.8565006256103516,"away_away_form_10":1.3425629138946533,"home_form_trend":1.7684166431427002,"away_form_trend":2.179579734802246,"form_diff":2.3272740840911865
  ,"home_xg_for_avg_3":2.046149730682373,"home_xg_against_avg_3":2.0002989768981934,"home_ppda_for_avg_3":2.0866856575012207,"home_ppda_against_avg_3":2.0045578479766846,"home_deep_comple
  tions_for_avg_3":2.0121335983276367,"home_deep_completions_against_avg_3":2.0322813987731934,"home_enriched_match_coverage_3":2.114503860473633,"home_xg_for_avg_5":1.8989803791046143,"h
  ome_xg_against_avg_5":2.097752332687378,"home_ppda_for_avg_5":2.31618070602417,"home_ppda_against_avg_5":2.0472381114959717,"home_deep_completions_for_avg_5":1.9418542385101318,"home_de
  ep_completions_against_avg_5":1.771629810333252,"home_enriched_match_coverage_5":1.183123230934143,"away_xg_for_avg_3":1.9834345579147339,"away_xg_against_avg_3":2.104243755340576,"away
  _ppda_for_avg_3":1.9828490018844604,"away_ppda_against_avg_3":2.2887394428253174,"away_deep_completions_for_avg_3":2.425812005996704,"away_deep_completions_against_avg_3":2.085261821746
  826,"away_enriched_match_coverage_3":2.478226661682129,"away_xg_for_avg_5":1.972304344177246,"away_xg_against_avg_5":2.199632167816162,"away_ppda_for_avg_5":2.088925838470459,"away_ppda
  _against_avg_5":1.9694545269012451,"away_deep_completions_for_avg_5":2.2806549072265625,"away_deep_completions_against_avg_5":2.441187620162964,"away_enriched_match_coverage_5":2.232986
  9270324707,"defensive_balance_3":2.1533501148223877,"low_scoring_probability_3":1.6657052040100098,"clean_sheet_interaction_3":1.5961768627166748,"goal_convergence_3":2.470282554626465,
  "volatility_sum_3":2.0173799991607666,"defensive_balance_5":2.1486990451812744,"low_scoring_probability_5":1.9356824159622192,"clean_sheet_interaction_5":1.2623343467712402,"goal_conver
  gence_5":2.5732502937316895,"volatility_sum_5":2.629129648208618,"defensive_balance_10":2.0297815799713135,"low_scoring_probability_10":2.4469850063323975,"clean_sheet_interaction_10":2
  .318861246109009,"goal_convergence_10":2.185242176055908,"volatility_sum_10":1.880535364151001,"xg_parity_3":2.2077412605285645,"xg_parity_5":1.9952961206436157,"strength_parity":2.1744
  26317214966,"h2h_draw_boost":1.2438099384307861,"strength_parity_5":2.836230754852295,"combined_defensive_strength_5":1.8641927242279053,"offensive_balance_5":1.9389632940292358,"total_
  goal_parity_5":2.2573158740997314,"low_scoring_matchup_5":1.9550752639770508,"win_rate_gap_5":2.4693796634674072,"combined_draw_tendency_5":1.3625658750534058,"strength_parity_10":2.079
  2295932769775,"combined_defensive_strength_10":1.8882278203964233,"offensive_balance_10":2.3349761962890625,"total_goal_parity_10":2.0375685691833496,"low_scoring_matchup_10":2.20669364
  9291992,"win_rate_gap_10":2.023331880569458,"combined_draw_tendency_10":2.1445703506469727,"day_of_week":1.9514449834823608,"month":2.0310752391815186,"is_weekend":0.0,"days_from_season
  _start":2.0337255001068115,"season_progress":2.033888578414917,"is_season_start":1.51503586769104,"is_season_mid":1.390419363975525,"is_season_late":0.0,"is_season_end":0.0,"home_rest_d
  ays":1.9657812118530273,"away_rest_days":1.8347517251968384,"rest_days_diff":1.8267576694488525,"home_matches_last_14_days":2.6482696533203125,"away_matches_last_14_days":0.0,"home_matc
  hes_last_7_days":1.7124004364013672,"away_matches_last_7_days":2.1517765522003174,"congestion_diff":2.2435648441314697,"congestion_symmetry":0.0,"rest_symmetry":1.7675577402114868},"ens
  emble_weights":null,"ensemble_validation_metrics":null,"ensemble_types":null}%
