# 🚪❄️🌡️ TON_IoT Multi-Device Machine Learning Pipeline

**Group 2 — IoT Intrusion Detection Using Machine Learning**

**Authors & Model Ownership**

* **Gary Mullings** — k-Nearest Neighbors (kNN)
* **Cierra Christian** — Random Forest
* **Megan Geer** — XGBoost

---

## 📌 Project Overview

The rapid expansion of Internet-of-Things (IoT) devices has created complex, high-volume telemetry streams that are difficult to secure using traditional rule-based methods. Each connected device introduces unique behavioral patterns and potential attack vectors, increasing the risk of undetected intrusions.

This project applies **machine learning–based intrusion detection** to **multiple IoT device types** using the **TON_IoT dataset**. A **single, consistent pipeline** is used across all devices to ensure fair comparison between models and datasets.

We evaluate three machine learning models:

* kNN
* Random Forest
* XGBoost

Across three IoT devices:

* Garage Door
* Fridge
* Thermostat

Each model is evaluated in **baseline** and **tuned** configurations.

---

## 🧠 Pipeline Design (Shared Across All Devices)

All datasets follow the same structure:

1. **Data Loading & Cleaning**
2. **Feature Engineering (time-based features)**
3. **Stratified Sampling (15%)**
4. **Train/Test Split (80/20)**
5. **Shared Preprocessing Recipe**
6. **Baseline Model Training**
7. **Model Evaluation**
8. **Hyperparameter Tuning (per model owner)**
9. **Final Comparison (Baseline vs Tuned)**

This ensures that performance differences reflect **model behavior**, not preprocessing inconsistencies.

---

## 🛠️ Libraries Used

```r
library(tidyverse)
library(tidymodels)
library(lubridate)
library(hms)
library(janitor)
library(kknn)
library(ranger)
library(xgboost)
library(themis)
library(finetune)
library(doParallel)
```

---

# 🚪 DATASET 1: GARAGE DOOR

### File

`IoT_Garage_Door.csv`

### Device-Specific Feature

* `sphone_signal` (categorical)

---

### 📄 Garage Door Pipeline Code

```r
```r
##############################################
# TON_IoT Garage Door Dataset - ML Pipeline
# Group 2: Cierra Christian, Megan Geer, Gary Mullings
# Ownership: Gary = kNN | Cierra = Random Forest | Megan = XGBoost
##############################################

##############################################
# Libraries
##############################################
library(tidyverse)
library(tidymodels)
library(lubridate)
library(hms)
library(janitor)
library(kknn)
library(ranger)
library(xgboost)
library(themis)
library(finetune)
library(doParallel)
set.seed(42)

##############################################
# 1. Load Dataset
##############################################
df_full <- read_csv("C:/Users/germe/OneDrive/Documents/R Code/IoT_Garage_Door.csv",
                    show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(
    date = as.Date(date, format = "%d-%b-%y"),
    day_of_week = wday(date),
    day_of_month = mday(date),
    is_weekend = if_else(day_of_week %in% c(1, 7), 1, 0),
    hour = hour(time),
    minute = minute(time),
    second = second(time),
    sphone_signal = as.factor(sphone_signal),
    label = factor(label, levels = c("1", "0"))
  ) %>%
  select(-date, -time, -type)

##############################################
# 1B. SAMPLE THE DATASET (STRATIFIED)
##############################################
df <- df_full %>%
  group_by(label) %>%
  sample_frac(0.15) %>%      # 15% of each class
  ungroup()

cat("Sampled rows:", nrow(df), "\n")
table(df$label)

##############################################
# 2. Train/Test Split
##############################################
split <- initial_split(df, prop = 0.80, strata = label)
train_df <- training(split)
test_df  <- testing(split)

##############################################
# 3. Preprocessing Recipe (shared across models)
##############################################
recipe_base <- recipe(label ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(label)

##############################################
# 4. BASELINE MODELS (kNN + RF + XGB)
##############################################

# ---------------------------
# Gary Mullings — kNN (Baseline)
# ---------------------------
knn_spec <- nearest_neighbor(
  mode = "classification",
  neighbors = 5,
  weight_func = "rectangular",
  dist_power = 2
) %>% 
  set_engine("kknn")

# ---------------------------
# Cierra Christian — Random Forest (Baseline)
# ---------------------------
rf_spec <- rand_forest(
  mode = "classification",
  trees = 500
) %>% 
  set_engine("ranger", probability = TRUE)

# ---------------------------
# Megan Geer — XGBoost (Baseline)
# ---------------------------
xgb_spec <- boost_tree(
  mode = "classification",
  trees = 200,
  tree_depth = 6,
  learn_rate = 0.1,
  loss_reduction = 0.01
) %>% set_engine("xgboost")

##############################################
# 5. BASELINE WORKFLOWS + FITS
##############################################
knn_wf <- workflow() %>% add_model(knn_spec) %>% add_recipe(recipe_base)
rf_wf  <- workflow() %>% add_model(rf_spec)  %>% add_recipe(recipe_base)
xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(recipe_base)

knn_fit <- fit(knn_wf, data = train_df)
rf_fit  <- fit(rf_wf,  data = train_df)
xgb_fit <- fit(xgb_wf, data = train_df)

##############################################
# 6. Megan — Feature Importance for Original XGBoost
##############################################
recipe_prepped <- prep(recipe_base)
xgb_matrix <- bake(recipe_prepped, new_data = train_df, all_predictors(), composition = "matrix")
xgb_engine <- extract_fit_engine(xgb_fit)
xgb_importance_tbl <- xgboost::xgb.importance(
  feature_names = colnames(xgb_matrix),
  model = xgb_engine
)

xgb_importance_plot <- ggplot(xgb_importance_tbl, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Original XGBoost Feature Importance – IoT_Garage_Door", x = "Feature", y = "Gain") +
  theme_minimal(base_size = 14)
print(xgb_importance_plot)

##############################################
# 7. Evaluation Helpers (shared)
##############################################
evaluate_model <- function(model_fit, test_data, model_name) {
  
  pred_df <- bind_cols(
    predict(model_fit, test_data),
    predict(model_fit, test_data, type = "prob"),
    test_data
  )
  
  metrics_tbl <- pred_df %>% metrics(truth = label, estimate = .pred_class)
  roc_auc_tbl <- pred_df %>% roc_auc(truth = label, .pred_1)
  pr_auc_tbl  <- pred_df %>% pr_auc(truth = label, .pred_1)
  f1_tbl      <- pred_df %>% f_meas(truth = label, estimate = .pred_class)
  cm          <- pred_df %>% conf_mat(truth = label, estimate = .pred_class)
  
  list(
    metrics  = bind_rows(metrics_tbl, roc_auc_tbl, pr_auc_tbl, f1_tbl),
    conf_mat = cm,
    pred     = pred_df
  )
}

plot_confusion_matrix <- function(conf_mat_obj, model_name) {
  
  cm_df <- as.data.frame(conf_mat_obj$table)
  colnames(cm_df) <- c("Truth", "Prediction", "Count")
  
  cm_df$Label <- NA
  cm_df$Label[cm_df$Truth == "1" & cm_df$Prediction == "1"] <- "True Positive"
  cm_df$Label[cm_df$Truth == "0" & cm_df$Prediction == "0"] <- "True Negative"
  cm_df$Label[cm_df$Truth == "0" & cm_df$Prediction == "1"] <- "False Positive"
  cm_df$Label[cm_df$Truth == "1" & cm_df$Prediction == "0"] <- "False Negative"
  
  cm_df$Percent <- round(cm_df$Count / sum(cm_df$Count) * 100, 2)
  
  ggplot(cm_df, aes(Prediction, Truth, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = paste0(Label, "\n", Count, "\n(", Percent, "%)")),
              fontface = "bold", size = 4) +
    scale_fill_gradient(low = "#F7FBFF", high = "orange") +
    labs(
      title = paste(model_name, "— Confusion Matrix"),
      x = "Predicted Class",
      y = "Actual Class",
      fill = "Count"
    ) +
    theme_minimal(base_size = 13)
}

##############################################
# 8. BASELINE RESULTS (All 3 models)
##############################################
knn_results <- evaluate_model(knn_fit, test_df, "kNN (Baseline — Gary)")
rf_results  <- evaluate_model(rf_fit,  test_df, "Random Forest (Baseline — Cierra)")
xgb_results <- evaluate_model(xgb_fit, test_df, "XGBoost (Baseline — Megan)")

print(knn_results$metrics); print(knn_results$conf_mat)
print(rf_results$metrics);  print(rf_results$conf_mat)
print(xgb_results$metrics); print(xgb_results$conf_mat)

plot_confusion_matrix(knn_results$conf_mat, "kNN (Baseline — Gary)")
plot_confusion_matrix(rf_results$conf_mat,  "Random Forest (Baseline — Cierra)")
plot_confusion_matrix(xgb_results$conf_mat, "XGBoost (Baseline — Megan)")

##############################################
# 9. ROC CURVES (Baseline comparison)
##############################################
roc_all <- bind_rows(
  knn_results$pred %>% 
    select(label, .pred_1) %>% 
    rename(truth = label, prob = .pred_1) %>%
    mutate(model = "kNN (Gary)"),
  
  rf_results$pred %>% 
    select(label, .pred_1) %>% 
    rename(truth = label, prob = .pred_1) %>%
    mutate(model = "Random Forest (Cierra)"),
  
  xgb_results$pred %>% 
    select(label, .pred_1) %>% 
    rename(truth = label, prob = .pred_1) %>% 
    mutate(model = "XGBoost (Megan)")
)

roc_curve_df <- roc_all %>%
  group_by(model) %>%
  roc_curve(truth, prob)

ggplot(roc_curve_df, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(size = 1.3) +
  geom_abline(lty = 3) +
  labs(
    title = "ROC Curve – Baseline Model Comparison (Garage Door)",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 14)

roc_all %>% group_by(model) %>% roc_auc(truth, prob)

#####################################################################
# 10. TUNED MODELS (ONE PER PERSON)
#####################################################################

##############################################
# 10A. Gary Mullings — Tuned kNN (Tune k)
##############################################
knn_tune_spec <- nearest_neighbor(
  mode        = "classification",
  neighbors   = tune(),
  weight_func = "rectangular",
  dist_power  = 2
) %>% set_engine("kknn")

knn_tune_wf <- workflow() %>%
  add_model(knn_tune_spec) %>%
  add_recipe(recipe_base)

folds   <- vfold_cv(train_df, v = 5, strata = label)
k_grid  <- tibble(neighbors = seq(1, 31, by = 2))
metric_k <- metric_set(accuracy, f_meas)

knn_tuned <- knn_tune_wf %>%
  tune_grid(
    resamples = folds,
    grid      = k_grid,
    metrics   = metric_k
  )

knn_tune_results <- knn_tuned %>% collect_metrics()

best_k <- knn_tune_results %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::arrange(desc(mean)) %>%
  dplyr::slice(1) %>%
  dplyr::pull(neighbors)

cat("Best k (Gary) based on CV accuracy:", best_k, "\n")

knn_spec_final <- nearest_neighbor(
  mode        = "classification",
  neighbors   = best_k,
  weight_func = "rectangular",
  dist_power  = 2
) %>% set_engine("kknn")

knn_final_wf  <- workflow() %>% add_model(knn_spec_final) %>% add_recipe(recipe_base)
knn_final_fit <- fit(knn_final_wf, data = train_df)

knn_tuned_results <- evaluate_model(knn_final_fit, test_df, "kNN (Tuned — Gary)")
print(knn_tuned_results$metrics)
print(knn_tuned_results$conf_mat)
plot_confusion_matrix(knn_tuned_results$conf_mat, "kNN (Tuned — Gary)")

##############################################
# 10B. Megan Geer — Tuned XGBoost (Bayesian)
##############################################
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

xgb_tune_wf <- workflow() %>%
  add_model(
    boost_tree(
      mode = "classification",
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      loss_reduction = tune()
    ) %>% set_engine("xgboost")
  ) %>%
  add_recipe(recipe_base)

xgb_resamples <- vfold_cv(train_df, v = 3, strata = label)

xgb_param_set <- parameters(
  trees(range = c(50, 150)),
  tree_depth(range = c(3, 6)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0))
)

set.seed(42)
xgb_bayes_results <- tune_bayes(
  xgb_tune_wf,
  resamples = xgb_resamples,
  param_info = xgb_param_set,
  initial = 6,
  iter = 4,
  metrics = metric_set(roc_auc),
  control = control_bayes(verbose = TRUE, save_pred = TRUE)
)

best_params <- select_best(xgb_bayes_results, metric ="roc_auc")
print(best_params)

xgb_final_wf  <- finalize_workflow(xgb_tune_wf, best_params)
xgb_final_fit <- fit(xgb_final_wf, data = train_df)

xgb_tuned_results <- evaluate_model(xgb_final_fit, test_df, "XGBoost (Tuned — Megan)")

stopCluster(cl)
registerDoSEQ()

print(xgb_tuned_results$metrics)
print(xgb_tuned_results$conf_mat)
plot_confusion_matrix(xgb_tuned_results$conf_mat, "XGBoost (Tuned — Megan)")

##############################################
# 10C. Megan — Feature Importance for Tuned XGBoost
##############################################
recipe_prepped_tuned <- prep(recipe_base)
xgb_matrix_tuned <- bake(recipe_prepped_tuned, new_data = train_df, all_predictors(), composition = "matrix")

xgb_tuned_importance_tbl <- xgboost::xgb.importance(
  feature_names = colnames(xgb_matrix_tuned),
  model = extract_fit_engine(xgb_final_fit)
)

xgb_tuned_plot <- ggplot(xgb_tuned_importance_tbl, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(title = "Tuned XGBoost Feature Importance – IoT_Garage_Door", x = "Feature", y = "Gain") +
  theme_minimal(base_size = 14)

print(xgb_tuned_plot)

##############################################
# 10D. Cierra Christian — Tuned Random Forest (Grid Search)
##############################################
rf_tune_spec <- rand_forest(
  mode  = "classification",
  trees = tune(),
  mtry  = tune(),
  min_n = tune()
) %>%
  set_engine(
    "ranger",
    probability = TRUE,
    importance   = "impurity"
  )

rf_tune_wf <- workflow() %>%
  add_model(rf_tune_spec) %>%
  add_recipe(recipe_base)

set.seed(42)
rf_resamples <- vfold_cv(train_df, v = 3, strata = label)

rf_param_set <- parameters(
  trees(range = c(200L, 800L)),
  mtry(range  = c(2L, 10L)),
  min_n(range = c(2L, 10L))
)

rf_tune_results <- tune_grid(
  rf_tune_wf,
  resamples  = rf_resamples,
  grid       = 15,
  metrics    = metric_set(roc_auc),
  param_info = rf_param_set
)

best_rf_params <- select_best(rf_tune_results, metric = "roc_auc")
cat("\n=== Best Tuned RF Parameters (ROC AUC) – Garage Door (Cierra) ===\n")
print(best_rf_params)

rf_final_wf  <- finalize_workflow(rf_tune_wf, best_rf_params)
rf_tuned_fit <- fit(rf_final_wf, data = train_df)

rf_tuned_results <- evaluate_model(rf_tuned_fit, test_df, "Random Forest (Tuned — Cierra)")

print(rf_tuned_results$metrics)
print(rf_tuned_results$conf_mat)
plot_confusion_matrix(rf_tuned_results$conf_mat, "Random Forest (Tuned — Garage Door — Cierra)")

##############################################
# 10E. Cierra — Tuned Random Forest Feature Importance & ROC
##############################################
rf_tuned_engine <- extract_fit_engine(rf_tuned_fit)

rf_tuned_importance_tbl <- enframe(
  rf_tuned_engine$variable.importance,
  name  = "Feature",
  value = "Importance"
) %>%
  arrange(desc(Importance))

ggplot(rf_tuned_importance_tbl,
       aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Tuned Random Forest Feature Importance – IoT_Garage_Door",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14)

rf_roc_tuned_df <- rf_tuned_results$pred %>%
  roc_curve(truth = label, .pred_1)

autoplot(rf_roc_tuned_df) +
  ggtitle("ROC Curve – Tuned Random Forest (IoT_Garage_Door)") +
  theme_minimal(base_size = 14)

##############################################
# 11. FINAL COMPARISON (Baseline vs Tuned per person)
##############################################
get_metrics_summary <- function(results_obj, model_name) {
  tibble(
    model     = model_name,
    accuracy  = accuracy(results_obj$pred, truth = label, estimate = .pred_class)[[".estimate"]],
    precision = precision(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    recall    = recall(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    f1        = f_meas(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    roc_auc   = roc_auc(results_obj$pred, truth = label, .pred_1)[[".estimate"]]
  )
}

garage_door_models_metrics <- bind_rows(
  get_metrics_summary(knn_results,       "kNN Baseline (Gary)"),
  get_metrics_summary(knn_tuned_results, "kNN Tuned (Gary)"),
  get_metrics_summary(xgb_results,       "XGBoost Baseline (Megan)"),
  get_metrics_summary(xgb_tuned_results, "XGBoost Tuned (Megan)"),
  get_metrics_summary(rf_results,        "Random Forest Baseline (Cierra)"),
  get_metrics_summary(rf_tuned_results,  "Random Forest Tuned (Cierra)")
)

cat("\n=== Garage Door – Baseline vs Tuned (One Per Person) ===\n")
print(garage_door_models_metrics)

##############################################
# END OF PIPELINE
##############################################
```

```

---

# ❄️ DATASET 2: FRIDGE

### File

`IoT_Fridge.csv`

### Device-Specific Feature

* `temp_condition` (categorical)

---

### 📄 Fridge Pipeline Code

```r
##############################################
# TON_IoT Fridge Dataset - ML Pipeline
# Group 2: Cierra Christian, Megan Geer, Gary Mullings
# Ownership: Gary = kNN | Cierra = Random Forest | Megan = XGBoost
##############################################

##############################################
# Libraries
##############################################
library(tidyverse)
library(tidymodels)
library(lubridate)
library(hms)
library(janitor)
library(kknn)
library(ranger)
library(xgboost)
library(themis)
library(finetune)
library(doParallel)
set.seed(42)

##############################################
# 1. Load Dataset
##############################################
df_full <- read_csv("C:/Users/germe/OneDrive/Documents/R Code/IoT_Fridge.csv",
                    show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(
    date = as.Date(date, format = "%d-%b-%y"),
    day_of_week = wday(date),
    day_of_month = mday(date),
    is_weekend = if_else(day_of_week %in% c(1, 7), 1, 0),
    hour = hour(time),
    minute = minute(time),
    second = second(time),
    temp_condition = as.factor(temp_condition),
    label = factor(label, levels = c("1", "0"))
  ) %>%
  select(-date, -time, -type)

##############################################
# 1B. SAMPLE THE DATASET (STRATIFIED)
##############################################
df <- df_full %>%
  group_by(label) %>%
  sample_frac(0.15) %>%      # 15% of each class
  ungroup()

cat("Sampled rows:", nrow(df), "\n")
table(df$label)

##############################################
# 2. Train/Test Split
##############################################
split <- initial_split(df, prop = 0.80, strata = label)
train_df <- training(split)
test_df  <- testing(split)

##############################################
# 3. Preprocessing Recipe (shared across models)
##############################################
recipe_base <- recipe(label ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(label)

##############################################
# 4. BASELINE MODELS (kNN + RF + XGB)
##############################################

# ---------------------------
# Gary Mullings — kNN (Baseline)
# ---------------------------
knn_spec <- nearest_neighbor(
  mode = "classification",
  neighbors = 5,
  weight_func = "rectangular",
  dist_power = 2
) %>% 
  set_engine("kknn")

# ---------------------------
# Cierra Christian — Random Forest (Baseline)
# ---------------------------
rf_spec <- rand_forest(
  mode = "classification",
  trees = 500
) %>% 
  set_engine("ranger", probability = TRUE)

# ---------------------------
# Megan Geer — XGBoost (Baseline)
# ---------------------------
xgb_spec <- boost_tree(
  mode = "classification",
  trees = 200,
  tree_depth = 6,
  learn_rate = 0.1,
  loss_reduction = 0.01
) %>% set_engine("xgboost")

##############################################
# 5. BASELINE WORKFLOWS + FITS
##############################################
knn_wf <- workflow() %>% add_model(knn_spec) %>% add_recipe(recipe_base)
rf_wf  <- workflow() %>% add_model(rf_spec)  %>% add_recipe(recipe_base)
xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(recipe_base)

knn_fit <- fit(knn_wf, data = train_df)
rf_fit  <- fit(rf_wf,  data = train_df)
xgb_fit <- fit(xgb_wf, data = train_df)

##############################################
# 6. Megan — Feature Importance for Original XGBoost
##############################################
recipe_prepped <- prep(recipe_base)
xgb_matrix <- bake(recipe_prepped, new_data = train_df, all_predictors(), composition = "matrix")
xgb_engine <- extract_fit_engine(xgb_fit)
xgb_importance_tbl <- xgboost::xgb.importance(
  feature_names = colnames(xgb_matrix),
  model = xgb_engine
)

xgb_importance_plot <- ggplot(xgb_importance_tbl, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Original XGBoost Feature Importance – IoT_Fridge", x = "Feature", y = "Gain") +
  theme_minimal(base_size = 14)
print(xgb_importance_plot)

##############################################
# 7. Evaluation Helpers (shared)
##############################################
evaluate_model <- function(model_fit, test_data, model_name) {
  
  pred_df <- bind_cols(
    predict(model_fit, test_data),
    predict(model_fit, test_data, type = "prob"),
    test_data
  )
  
  metrics_tbl <- pred_df %>% metrics(truth = label, estimate = .pred_class)
  roc_auc_tbl <- pred_df %>% roc_auc(truth = label, .pred_1)
  pr_auc_tbl  <- pred_df %>% pr_auc(truth = label, .pred_1)
  f1_tbl      <- pred_df %>% f_meas(truth = label, estimate = .pred_class)
  cm          <- pred_df %>% conf_mat(truth = label, estimate = .pred_class)
  
  list(
    metrics  = bind_rows(metrics_tbl, roc_auc_tbl, pr_auc_tbl, f1_tbl),
    conf_mat = cm,
    pred     = pred_df
  )
}

plot_confusion_matrix <- function(conf_mat_obj, model_name) {
  
  cm_df <- as.data.frame(conf_mat_obj$table)
  colnames(cm_df) <- c("Truth", "Prediction", "Count")
  
  cm_df$Label <- NA
  cm_df$Label[cm_df$Truth == "1" & cm_df$Prediction == "1"] <- "True Positive"
  cm_df$Label[cm_df$Truth == "0" & cm_df$Prediction == "0"] <- "True Negative"
  cm_df$Label[cm_df$Truth == "0" & cm_df$Prediction == "1"] <- "False Positive"
  cm_df$Label[cm_df$Truth == "1" & cm_df$Prediction == "0"] <- "False Negative"
  
  cm_df$Percent <- round(cm_df$Count / sum(cm_df$Count) * 100, 2)
  
  ggplot(cm_df, aes(Prediction, Truth, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = paste0(Label, "\n", Count, "\n(", Percent, "%)")),
              fontface = "bold", size = 4) +
    scale_fill_gradient(low = "#F7FBFF", high = "orange") +
    labs(
      title = paste(model_name, "— Confusion Matrix"),
      x = "Predicted Class",
      y = "Actual Class",
      fill = "Count"
    ) +
    theme_minimal(base_size = 13)
}

##############################################
# 8. BASELINE RESULTS (All 3 models)
##############################################
knn_results <- evaluate_model(knn_fit, test_df, "kNN (Baseline — Gary)")
rf_results  <- evaluate_model(rf_fit,  test_df, "Random Forest (Baseline — Cierra)")
xgb_results <- evaluate_model(xgb_fit, test_df, "XGBoost (Baseline — Megan)")

print(knn_results$metrics); print(knn_results$conf_mat)
print(rf_results$metrics);  print(rf_results$conf_mat)
print(xgb_results$metrics); print(xgb_results$conf_mat)

plot_confusion_matrix(knn_results$conf_mat, "kNN (Baseline — Gary)")
plot_confusion_matrix(rf_results$conf_mat,  "Random Forest (Baseline — Cierra)")
plot_confusion_matrix(xgb_results$conf_mat, "XGBoost (Baseline — Megan)")

##############################################
# 9. ROC CURVES (Baseline comparison)
##############################################
roc_all <- bind_rows(
  knn_results$pred %>% 
    select(label, .pred_1) %>% 
    rename(truth = label, prob = .pred_1) %>%
    mutate(model = "kNN (Gary)"),
  
  rf_results$pred %>% 
    select(label, .pred_1) %>% 
    rename(truth = label, prob = .pred_1) %>%
    mutate(model = "Random Forest (Cierra)"),
  
  xgb_results$pred %>% 
    select(label, .pred_1) %>% 
    rename(truth = label, prob = .pred_1) %>% 
    mutate(model = "XGBoost (Megan)")
)

roc_curve_df <- roc_all %>%
  group_by(model) %>%
  roc_curve(truth, prob)

ggplot(roc_curve_df, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(size = 1.3) +
  geom_abline(lty = 3) +
  labs(
    title = "ROC Curve – Baseline Model Comparison (Fridge)",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 14)

roc_all %>% group_by(model) %>% roc_auc(truth, prob)

#####################################################################
# 10. TUNED MODELS (ONE PER PERSON)
#####################################################################

##############################################
# 10A. Gary Mullings — Tuned kNN (Tune k)
##############################################
knn_tune_spec <- nearest_neighbor(
  mode        = "classification",
  neighbors   = tune(),
  weight_func = "rectangular",
  dist_power  = 2
) %>% set_engine("kknn")

knn_tune_wf <- workflow() %>%
  add_model(knn_tune_spec) %>%
  add_recipe(recipe_base)

folds   <- vfold_cv(train_df, v = 5, strata = label)
k_grid  <- tibble(neighbors = seq(1, 31, by = 2))
metric_k <- metric_set(accuracy, f_meas)

knn_tuned <- knn_tune_wf %>%
  tune_grid(
    resamples = folds,
    grid      = k_grid,
    metrics   = metric_k
  )

knn_tune_results <- knn_tuned %>% collect_metrics()

best_k <- knn_tune_results %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::arrange(desc(mean)) %>%
  dplyr::slice(1) %>%
  dplyr::pull(neighbors)

cat("Best k (Gary) based on CV accuracy:", best_k, "\n")

knn_spec_final <- nearest_neighbor(
  mode        = "classification",
  neighbors   = best_k,
  weight_func = "rectangular",
  dist_power  = 2
) %>% set_engine("kknn")

knn_final_wf  <- workflow() %>% add_model(knn_spec_final) %>% add_recipe(recipe_base)
knn_final_fit <- fit(knn_final_wf, data = train_df)

knn_tuned_results <- evaluate_model(knn_final_fit, test_df, "kNN (Tuned — Gary)")
print(knn_tuned_results$metrics)
print(knn_tuned_results$conf_mat)
plot_confusion_matrix(knn_tuned_results$conf_mat, "kNN (Tuned — Gary)")

##############################################
# 10B. Megan Geer — Tuned XGBoost (Bayesian)
##############################################
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

xgb_tune_wf <- workflow() %>%
  add_model(
    boost_tree(
      mode = "classification",
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      loss_reduction = tune()
    ) %>% set_engine("xgboost")
  ) %>%
  add_recipe(recipe_base)

xgb_resamples <- vfold_cv(train_df, v = 3, strata = label)

xgb_param_set <- parameters(
  trees(range = c(50, 150)),
  tree_depth(range = c(3, 6)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0))
)

set.seed(42)
xgb_bayes_results <- tune_bayes(
  xgb_tune_wf,
  resamples = xgb_resamples,
  param_info = xgb_param_set,
  initial = 6,
  iter = 4,
  metrics = metric_set(roc_auc),
  control = control_bayes(verbose = TRUE, save_pred = TRUE)
)

best_params <- select_best(xgb_bayes_results, metric ="roc_auc")
print(best_params)

xgb_final_wf  <- finalize_workflow(xgb_tune_wf, best_params)
xgb_final_fit <- fit(xgb_final_wf, data = train_df)

xgb_tuned_results <- evaluate_model(xgb_final_fit, test_df, "XGBoost (Tuned — Megan)")

stopCluster(cl)
registerDoSEQ()

print(xgb_tuned_results$metrics)
print(xgb_tuned_results$conf_mat)
plot_confusion_matrix(xgb_tuned_results$conf_mat, "XGBoost (Tuned — Megan)")

##############################################
# 10C. Megan — Feature Importance for Tuned XGBoost
##############################################
recipe_prepped_tuned <- prep(recipe_base)
xgb_matrix_tuned <- bake(recipe_prepped_tuned, new_data = train_df, all_predictors(), composition = "matrix")

xgb_tuned_importance_tbl <- xgboost::xgb.importance(
  feature_names = colnames(xgb_matrix_tuned),
  model = extract_fit_engine(xgb_final_fit)
)

xgb_tuned_plot <- ggplot(xgb_tuned_importance_tbl, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "darkorange") +
  coord_flip() +
  labs(title = "Tuned XGBoost Feature Importance – IoT_Fridge", x = "Feature", y = "Gain") +
  theme_minimal(base_size = 14)

print(xgb_tuned_plot)

##############################################
# 10D. Cierra Christian — Tuned Random Forest (Grid Search)
##############################################
rf_tune_spec <- rand_forest(
  mode  = "classification",
  trees = tune(),
  mtry  = tune(),
  min_n = tune()
) %>%
  set_engine(
    "ranger",
    probability = TRUE,
    importance   = "impurity"
  )

rf_tune_wf <- workflow() %>%
  add_model(rf_tune_spec) %>%
  add_recipe(recipe_base)

set.seed(42)
rf_resamples <- vfold_cv(train_df, v = 3, strata = label)

rf_param_set <- parameters(
  trees(range = c(200L, 800L)),
  mtry(range  = c(2L, 10L)),
  min_n(range = c(2L, 10L))
)

rf_tune_results <- tune_grid(
  rf_tune_wf,
  resamples  = rf_resamples,
  grid       = 15,
  metrics    = metric_set(roc_auc),
  param_info = rf_param_set
)

best_rf_params <- select_best(rf_tune_results, metric = "roc_auc")
cat("\n=== Best Tuned RF Parameters (ROC AUC) – Fridge (Cierra) ===\n")
print(best_rf_params)

rf_final_wf  <- finalize_workflow(rf_tune_wf, best_rf_params)
rf_tuned_fit <- fit(rf_final_wf, data = train_df)

rf_tuned_results <- evaluate_model(rf_tuned_fit, test_df, "Random Forest (Tuned — Cierra)")

print(rf_tuned_results$metrics)
print(rf_tuned_results$conf_mat)
plot_confusion_matrix(rf_tuned_results$conf_mat, "Random Forest (Tuned — Fridge — Cierra)")

##############################################
# 10E. Cierra — Tuned Random Forest Feature Importance & ROC
##############################################
rf_tuned_engine <- extract_fit_engine(rf_tuned_fit)

rf_tuned_importance_tbl <- enframe(
  rf_tuned_engine$variable.importance,
  name  = "Feature",
  value = "Importance"
) %>%
  arrange(desc(Importance))

ggplot(rf_tuned_importance_tbl,
       aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Tuned Random Forest Feature Importance – IoT_Fridge",
    x = "Feature",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14)

rf_roc_tuned_df <- rf_tuned_results$pred %>%
  roc_curve(truth = label, .pred_1)

autoplot(rf_roc_tuned_df) +
  ggtitle("ROC Curve – Tuned Random Forest (IoT_Fridge)") +
  theme_minimal(base_size = 14)

##############################################
# 11. FINAL COMPARISON (Baseline vs Tuned per person)
##############################################
get_metrics_summary <- function(results_obj, model_name) {
  tibble(
    model     = model_name,
    accuracy  = accuracy(results_obj$pred, truth = label, estimate = .pred_class)[[".estimate"]],
    precision = precision(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    recall    = recall(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    f1        = f_meas(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    roc_auc   = roc_auc(results_obj$pred, truth = label, .pred_1)[[".estimate"]]
  )
}

fridge_models_metrics <- bind_rows(
  get_metrics_summary(knn_results,       "kNN Baseline (Gary)"),
  get_metrics_summary(knn_tuned_results, "kNN Tuned (Gary)"),
  get_metrics_summary(xgb_results,       "XGBoost Baseline (Megan)"),
  get_metrics_summary(xgb_tuned_results, "XGBoost Tuned (Megan)"),
  get_metrics_summary(rf_results,        "Random Forest Baseline (Cierra)"),
  get_metrics_summary(rf_tuned_results,  "Random Forest Tuned (Cierra)")
)

cat("\n=== Fridge – Baseline vs Tuned (One Per Person) ===\n")
print(fridge_models_metrics)

##############################################
# END OF PIPELINE
##############################################
```


---

# 🌡️ DATASET 3: THERMOSTAT

### File

`IoT_Thermostat.csv`

### Device-Specific Feature

* `thermostat_status` (categorical)

---

### 📄 Thermostat Pipeline Code

```r
```r
##############################################
# TON_IoT Thermostat Dataset - ML Pipeline
# Group 2: Cierra Christian, Megan Geer, Gary Mullings
##############################################

##############################################
# Libraries
##############################################
library(tidyverse)
library(tidymodels)
library(lubridate)
library(hms)
library(janitor)
library(kknn)
library(ranger)
library(xgboost)
library(themis)
library(finetune)
library(doParallel)
set.seed(42)

##############################################
# 1. Load Dataset (Thermostat)
##############################################
df_full <- read_csv("C:/Users/germe/OneDrive/Documents/R Code/IoT_Thermostat.csv",
                    show_col_types = FALSE) %>%
  clean_names() %>%
  mutate(
    date = as.Date(date, format = "%d-%b-%y"),
    day_of_week = wday(date),
    day_of_month = mday(date),
    is_weekend = if_else(day_of_week %in% c(1, 7), 1, 0),
    hour = hour(time),
    minute = minute(time),
    second = second(time),
    thermostat_status = as.factor(thermostat_status),
    label = factor(label, levels = c("1", "0"))
  ) %>%
  select(-date, -time, -type)

##############################################
# 1B. SAMPLE THE DATASET (STRATIFIED)
##############################################
df <- df_full %>%
  group_by(label) %>%
  sample_frac(0.15) %>%      # 15% of each class
  ungroup()

cat("Sampled rows:", nrow(df), "\n")
table(df$label)

##############################################
# 2. Train/Test Split
##############################################
split <- initial_split(df, prop = 0.80, strata = label)
train_df <- training(split)
test_df  <- testing(split)

##############################################
# 3. Preprocessing Recipe (shared across models)
##############################################
recipe_base <- recipe(label ~ ., data = train_df) %>%
  step_zv(all_predictors()) %>%
  step_impute_median(all_numeric()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(label)

##############################################
# 4. BASELINE MODELS (kNN + RF + XGB)
##############################################

# ---------------------------
# Gary Mullings — kNN (baseline spec used before tuning)
# ---------------------------
knn_spec <- nearest_neighbor(
  mode = "classification",
  neighbors = 5,
  weight_func = "rectangular",
  dist_power = 2
) %>% 
  set_engine("kknn")

# ---------------------------
# Cierra Christian — Random Forest (baseline)
# ---------------------------
rf_spec <- rand_forest(
  mode  = "classification",
  trees = 500
) %>% 
  set_engine("ranger", probability = TRUE)

# ---------------------------
# Megan Geer — XGBoost (baseline)
# ---------------------------
xgb_spec <- boost_tree(
  mode = "classification",
  trees = 200,
  tree_depth = 6,
  learn_rate = 0.1,
  loss_reduction = 0.01
) %>% set_engine("xgboost")

##############################################
# 5. BASELINE WORKFLOWS + FITS
##############################################
knn_wf <- workflow() %>% add_model(knn_spec) %>% add_recipe(recipe_base)
rf_wf  <- workflow() %>% add_model(rf_spec)  %>% add_recipe(recipe_base)
xgb_wf <- workflow() %>% add_model(xgb_spec) %>% add_recipe(recipe_base)

knn_fit <- fit(knn_wf, data = train_df)
rf_fit  <- fit(rf_wf,  data = train_df)
xgb_fit <- fit(xgb_wf, data = train_df)

##############################################
# 6. EVALUATION HELPERS (shared)
##############################################
evaluate_model <- function(model_fit, test_data, model_name) {
  
  pred_df <- bind_cols(
    predict(model_fit, test_data),
    predict(model_fit, test_data, type = "prob"),
    test_data
  )
  
  metrics_tbl <- pred_df %>% metrics(truth = label, estimate = .pred_class)
  roc_auc_tbl <- pred_df %>% roc_auc(truth = label, .pred_1)
  pr_auc_tbl  <- pred_df %>% pr_auc(truth = label, .pred_1)
  f1_tbl      <- pred_df %>% f_meas(truth = label, estimate = .pred_class)
  cm          <- pred_df %>% conf_mat(truth = label, estimate = .pred_class)
  
  list(
    metrics  = bind_rows(metrics_tbl, roc_auc_tbl, pr_auc_tbl, f1_tbl),
    conf_mat = cm,
    pred     = pred_df
  )
}

plot_confusion_matrix <- function(conf_mat_obj, model_name) {
  
  cm_df <- as.data.frame(conf_mat_obj$table)
  colnames(cm_df) <- c("Truth", "Prediction", "Count")
  
  cm_df$Label <- NA
  cm_df$Label[cm_df$Truth == "1" & cm_df$Prediction == "1"] <- "True Positive"
  cm_df$Label[cm_df$Truth == "0" & cm_df$Prediction == "0"] <- "True Negative"
  cm_df$Label[cm_df$Truth == "0" & cm_df$Prediction == "1"] <- "False Positive"
  cm_df$Label[cm_df$Truth == "1" & cm_df$Prediction == "0"] <- "False Negative"
  
  cm_df$Percent <- round(cm_df$Count / sum(cm_df$Count) * 100, 2)
  
  ggplot(cm_df, aes(Prediction, Truth, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = paste0(Label, "\n", Count, "\n(", Percent, "%)")),
              fontface = "bold", size = 4) +
    scale_fill_gradient(low = "#F7FBFF", high = "orange") +
    labs(
      title = paste(model_name, "— Confusion Matrix"),
      x = "Predicted Class",
      y = "Actual Class",
      fill = "Count"
    ) +
    theme_minimal(base_size = 13)
}

##############################################
# 7. BASELINE RESULTS (All 3 models)
##############################################
knn_results <- evaluate_model(knn_fit, test_df, "kNN (Baseline — Gary)")
rf_results  <- evaluate_model(rf_fit,  test_df, "Random Forest (Baseline — Cierra)")
xgb_results <- evaluate_model(xgb_fit, test_df, "XGBoost (Baseline — Megan)")

print(knn_results$metrics); print(knn_results$conf_mat)
print(rf_results$metrics);  print(rf_results$conf_mat)
print(xgb_results$metrics); print(xgb_results$conf_mat)

plot_confusion_matrix(knn_results$conf_mat, "kNN (Baseline — Gary)")
plot_confusion_matrix(rf_results$conf_mat,  "Random Forest (Baseline — Cierra)")
plot_confusion_matrix(xgb_results$conf_mat, "XGBoost (Baseline — Megan)")

##############################################
# 8. ROC CURVES (Baseline comparison)
##############################################
roc_all <- bind_rows(
  knn_results$pred %>% select(label, .pred_1) %>% rename(truth = label, prob = .pred_1) %>% mutate(model = "kNN (Gary)"),
  rf_results$pred  %>% select(label, .pred_1) %>% rename(truth = label, prob = .pred_1) %>% mutate(model = "Random Forest (Cierra)"),
  xgb_results$pred %>% select(label, .pred_1) %>% rename(truth = label, prob = .pred_1) %>% mutate(model = "XGBoost (Megan)")
)

roc_curve_df <- roc_all %>%
  group_by(model) %>%
  roc_curve(truth, prob)

ggplot(roc_curve_df, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_line(size = 1.3) +
  geom_abline(lty = 3) +
  labs(
    title = "ROC Curve – Baseline Model Comparison (Thermostat)",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_minimal(base_size = 14)

roc_all %>% group_by(model) %>% roc_auc(truth, prob)

#####################################################################
# 9. TUNED MODELS (ONE PER PERSON)
#####################################################################

##############################################
# 9A. Gary Mullings — Tuned kNN (Tune k)
##############################################
knn_tune_spec <- nearest_neighbor(
  mode        = "classification",
  neighbors   = tune(),
  weight_func = "rectangular",
  dist_power  = 2
) %>% 
  set_engine("kknn")

knn_tune_wf <- workflow() %>% add_model(knn_tune_spec) %>% add_recipe(recipe_base)

folds   <- vfold_cv(train_df, v = 5, strata = label)
k_grid  <- tibble(neighbors = seq(1, 31, by = 2))
metric_k <- metric_set(accuracy, f_meas)

knn_tuned <- knn_tune_wf %>%
  tune_grid(
    resamples = folds,
    grid      = k_grid,
    metrics   = metric_k
  )

knn_tune_results <- knn_tuned %>% collect_metrics()

best_k <- knn_tune_results %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::arrange(desc(mean)) %>%
  dplyr::slice(1) %>%
  dplyr::pull(neighbors)

cat("Best k (Gary) based on CV accuracy:", best_k, "\n")

knn_spec_final <- nearest_neighbor(
  mode        = "classification",
  neighbors   = best_k,
  weight_func = "rectangular",
  dist_power  = 2
) %>% 
  set_engine("kknn")

knn_final_wf  <- workflow() %>% add_model(knn_spec_final) %>% add_recipe(recipe_base)
knn_final_fit <- fit(knn_final_wf, data = train_df)
knn_tuned_results <- evaluate_model(knn_final_fit, test_df, "kNN (Tuned — Gary)")

print(knn_tuned_results$metrics)
print(knn_tuned_results$conf_mat)
plot_confusion_matrix(knn_tuned_results$conf_mat, "kNN (Tuned — Gary)")

##############################################
# 9B. Megan Geer — Tuned XGBoost (Bayesian)
##############################################
cl <- makeCluster(parallel::detectCores() - 1)
registerDoParallel(cl)

xgb_tune_wf <- workflow() %>%
  add_model(
    boost_tree(
      mode = "classification",
      trees = tune(),
      tree_depth = tune(),
      learn_rate = tune(),
      loss_reduction = tune()
    ) %>% set_engine("xgboost")
  ) %>%
  add_recipe(recipe_base)

xgb_resamples <- vfold_cv(train_df, v = 3, strata = label)

xgb_param_set <- parameters(
  trees(range = c(50, 150)),
  tree_depth(range = c(3, 6)),
  learn_rate(range = c(-3, -1)),
  loss_reduction(range = c(-3, 0))
)

set.seed(42)
xgb_bayes_results <- tune_bayes(
  xgb_tune_wf,
  resamples = xgb_resamples,
  param_info = xgb_param_set,
  initial = 6,
  iter = 4,
  metrics = metric_set(roc_auc),
  control = control_bayes(verbose = TRUE, save_pred = TRUE)
)

best_params <- select_best(xgb_bayes_results, metric ="roc_auc")
print(best_params)

xgb_final_wf  <- finalize_workflow(xgb_tune_wf, best_params)
xgb_final_fit <- fit(xgb_final_wf, data = train_df)

xgb_tuned_results <- evaluate_model(xgb_final_fit, test_df, "XGBoost (Tuned — Megan)")

stopCluster(cl)
registerDoSEQ()

print(xgb_tuned_results$metrics)
print(xgb_tuned_results$conf_mat)
plot_confusion_matrix(xgb_tuned_results$conf_mat, "XGBoost (Tuned — Megan)")

##############################################
# 9C. Cierra Christian — Tuned Random Forest (Grid Search)
##############################################
rf_tune_spec <- rand_forest(
  mode  = "classification",
  trees = tune(),
  mtry  = tune(),
  min_n = tune()
) %>%
  set_engine(
    "ranger",
    probability = TRUE,
    importance   = "impurity"
  )

rf_tune_wf <- workflow() %>%
  add_model(rf_tune_spec) %>%
  add_recipe(recipe_base)

set.seed(42)
rf_resamples <- vfold_cv(train_df, v = 3, strata = label)

rf_param_set <- parameters(
  trees(range = c(200L, 800L)),
  mtry(range  = c(2L, 10L)),
  min_n(range = c(2L, 10L))
)

rf_tune_results <- tune_grid(
  rf_tune_wf,
  resamples  = rf_resamples,
  grid       = 15,
  metrics    = metric_set(roc_auc),
  param_info = rf_param_set
)

best_rf_params <- select_best(rf_tune_results, metric = "roc_auc")
cat("\n=== Best Tuned RF Parameters (Cierra) ===\n")
print(best_rf_params)

rf_final_wf  <- finalize_workflow(rf_tune_wf, best_rf_params)
rf_tuned_fit <- fit(rf_final_wf, data = train_df)

rf_tuned_results <- evaluate_model(rf_tuned_fit, test_df, "Random Forest (Tuned — Cierra)")

print(rf_tuned_results$metrics)
print(rf_tuned_results$conf_mat)
plot_confusion_matrix(rf_tuned_results$conf_mat, "Random Forest (Tuned — Cierra)")

##############################################
# 10. FINAL COMPARISON (Baseline vs Tuned per person)
##############################################
# NOTE: This section keeps your same metric logic style.
# It reads clean for the report/presentation.

get_metrics_summary <- function(results_obj, model_name) {
  tibble(
    model     = model_name,
    accuracy  = accuracy(results_obj$pred, truth = label, estimate = .pred_class)[[".estimate"]],
    precision = precision(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    recall    = recall(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    f1        = f_meas(results_obj$pred, truth = label, estimate = .pred_class, event_level = "second")[[".estimate"]],
    roc_auc   = roc_auc(results_obj$pred, truth = label, .pred_1)[[".estimate"]]
  )
}

Thermostat_models_metrics <- bind_rows(
  get_metrics_summary(knn_results,       "kNN Baseline (Gary)"),
  get_metrics_summary(knn_tuned_results, "kNN Tuned (Gary)"),
  get_metrics_summary(xgb_results,       "XGBoost Baseline (Megan)"),
  get_metrics_summary(xgb_tuned_results, "XGBoost Tuned (Megan)"),
  get_metrics_summary(rf_results,        "Random Forest Baseline (Cierra)"),
  get_metrics_summary(rf_tuned_results,  "Random Forest Tuned (Cierra)")
)

cat("\n=== Thermostat – Baseline vs Tuned (One Per Person) ===\n")
print(Thermostat_models_metrics)

##############################################
# END OF PIPELINE
##############################################
```

```

---

## 🤖 Models (Shared Across All Datasets)

```r
# Gary — kNN
knn_spec <- nearest_neighbor(
  mode = "classification",
  neighbors = 5,
  weight_func = "rectangular",
  dist_power = 2
) %>% set_engine("kknn")

# Cierra — Random Forest
rf_spec <- rand_forest(
  mode = "classification",
  trees = 500
) %>% set_engine("ranger", probability = TRUE)

# Megan — XGBoost
xgb_spec <- boost_tree(
  mode = "classification",
  trees = 200,
  tree_depth = 6,
  learn_rate = 0.1,
  loss_reduction = 0.01
) %>% set_engine("xgboost")
```

---

## 📈 Evaluation Metrics

All models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Confusion Matrix
* ROC Curves

---

## 🔑 Key Findings (Summary)

* **Tuned XGBoost** achieved the highest ROC-AUC across devices
* **Random Forest** provided strong interpretability with competitive accuracy
* **kNN** benefited significantly from tuning but remained sensitive to noise
* Shared preprocessing ensured fair, apples-to-apples comparison

---

## 📂 Repository Structure

```text
.
├── Garage_Door_Pipeline.R
├── Fridge_Pipeline.R
├── Thermostat_Pipeline.R
└── README.md
```

---

## 🎓 Academic Context

This project demonstrates:

* Multi-device ML experimentation
* Proper experimental control
* Clear ownership attribution
* Interpretable security analytics

---
