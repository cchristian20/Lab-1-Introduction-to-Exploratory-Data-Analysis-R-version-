---
---
---

# 🧠 CYBR 520 – Lab 3: Supervised Machine Learning (R Version)

**Authors:** Cierra Christian, Megan Geer, and Gary Mullings\
**Course:** Data Analytics for Cybersecurity (CYBR 520)\
**Institution:** West Virginia University\
**Tools and Packages:** R, RStudio, tidymodels, kknn, kernlab, ranger, ggplot2, pROC, vip

------------------------------------------------------------------------

## 🎯 Overview

This lab explores **Supervised Machine Learning** using the **Spambase dataset** to classify emails as spam (1) or nonspam (0).\
We implemented and compared three models — **k-Nearest Neighbors (k-NN)**, **Support Vector Machines (SVM)**, and **Random Forest (RF)** — to determine which algorithm provides the best accuracy, recall, precision, and ROC-AUC for spam detection.

------------------------------------------------------------------------

## 🧩 Dataset: Spambase

-   Developed by Hewlett-Packard Labs (1999)\
-   4601 total email samples
    -   2788 labeled as **nonspam**
    -   1813 labeled as **spam**\
-   57 predictive features representing word frequencies and character statistics\
-   Goal: Train multiple classifiers to predict spam probability and evaluate performance metrics.

## 📦 1. Import and Prepare the Dataset

``` r
# Required Libraries
install.packages(c("tidymodels", "janitor", "pROC", "kknn", "kernlab", "ggplot2", "ranger", "vip"))
library(tidymodels)
library(janitor)
library(pROC)
library(kknn)
library(kernlab)
library(ggplot2)
library(ranger)
library(vip)

# Load and Clean Data
spambase <- read.csv("C:/Users/cchri/Downloads/spambase.csv") %>% clean_names()
spambase$type <- factor(spambase$type, levels = c("nonspam", "spam"))

glimpse(spambase)
```

**Answer (Q1–Q2):**\
- Total emails: 4601 (2788 nonspam, 1813 spam)\
- Setting factor levels ensures the model recognizes *nonspam* as negative (0) and *spam* as positive (1).

------------------------------------------------------------------------

## ✂️ 2. Partition the Dataset

``` r
set.seed(42)
split <- initial_split(spambase, prop = 0.8, strata = type)
train <- training(split)
test  <- testing(split)
folds <- vfold_cv(train, v = 5, strata = type)
```

**Answer (Q3):**\
We split data first to ensure an **unseen test set**, preventing **data leakage** during cross-validation.

------------------------------------------------------------------------

## ⚙️ 3. Preprocessing & Normalization

Normalization ensures each variable contributes equally to distance-based models like **k-NN** and **SVM**.

``` r
base_rec <- recipe(type ~ ., data = train) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors())
```

**Answer (Q4):**\
Normalization is critical for algorithms that rely on Euclidean distance, improving model fairness and convergence.

------------------------------------------------------------------------

## 🌱 4. k-Nearest Neighbors (k-NN)

``` r
knn_spec <- nearest_neighbor(neighbors = tune(), mode = "classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(knn_spec)

knn_grid <- tibble(neighbors = seq(1, 31, by = 2))

set.seed(42)
knn_res <- tune_grid(knn_wf, resamples = folds, grid = knn_grid,
                     metrics = metric_set(accuracy, precision, recall, roc_auc))

autoplot(knn_res, metric = "accuracy")
show_best(knn_res, metric = "accuracy")
```

**Answer (Q7):**\
Best **k = 3** with accuracy ≈ **0.911**.\
Smaller *k* values captured finer distinctions between spam and nonspam.

**Answer (Q8–Q9):**\
- False Positives: 59\
- False Negatives: 35\
- FP = real emails mislabeled as spam; FN = spam reaching inbox (more dangerous).

------------------------------------------------------------------------

## 🧠 5. Support Vector Machine (SVM – RBF Kernel)

``` r
svm_spec <- svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(svm_spec)

svm_grid <- grid_latin_hypercube(cost(), rbf_sigma(), size = 20)

set.seed(42)
svm_res <- tune_grid(svm_wf, resamples = folds, grid = svm_grid,
                     metrics = metric_set(accuracy, precision, recall, roc_auc))

autoplot(svm_res, metric = "accuracy")
show_best(svm_res, metric = "accuracy")
```

**Answer (Q10–Q12):**\
- Best parameters: **cost = 12.5**, **rbf_sigma = 0.00433**, accuracy = **0.934**\
- `cost` controls misclassification penalty; `rbf_sigma` controls decision boundary flexibility.

**Answer (Q13–Q15):**\
- ROC AUC = **0.9749**, indicating strong model performance.\
- Support vectors define decision margins; spam clusters corresponded to emails with aggressive capitalization.

------------------------------------------------------------------------

## 🌳 6. Random Forest (Exploration–Other ML Methods)

``` r
rf_spec <- rand_forest(trees = tune(), mtry = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(base_rec) %>%
  add_model(rf_spec)

rf_grid <- grid_latin_hypercube(trees(), mtry(), min_n(), size = 10)

set.seed(42)
rf_res <- tune_grid(rf_wf, resamples = folds, grid = rf_grid,
                    metrics = metric_set(accuracy, precision, recall, roc_auc))

show_best(rf_res, metric = "accuracy")
rf_fit <- finalize_workflow(rf_wf, select_best(rf_res, "accuracy")) %>% fit(train)

vip(extract_fit_parsnip(rf_fit), num_features = 15) +
  labs(title = "Random Forest – Top 15 Feature Importances")
```

**Answer (Q20):**\
Random Forest achieved **Accuracy = 0.955**, **Recall = 0.978**, and **ROC-AUC = 0.9822**, outperforming SVM and k-NN.\
It’s ideal for real-world spam filters due to efficiency, scalability, and low tuning complexity.

``` r
Code:
    library(ranger)
library(vip)

# Track runtime
rf_time <- system.time({
    
    # 1. Specify the model
    rf_spec <- rand_forest(
        trees = tune(),       # number of trees
        mtry  = tune(),       # number of variables randomly sampled as candidates at each split
        min_n = tune()        # minimum node size
    ) %>%
        set_engine("ranger", importance = "impurity") %>%
        set_mode("classification")
    
    # 2. Build a workflow using base_rec
    rf_wf <- workflow() %>%
        add_recipe(base_rec) %>%
        add_model(rf_spec)
    
    # 3. Create a tuning grid
    rf_grid <- grid_latin_hypercube(
        trees(range = c(200, 1000)),
        mtry(range = c(3, 15)),
        min_n(range = c(1, 10)),
        size = 10
    )
    
    # 4. Tune the model using 5-fold CV
    set.seed(42)
    rf_res <- tune_grid(
        rf_wf,
        resamples = folds,
        grid = rf_grid,
        metrics = eval_metrics,
        control = control_grid(save_pred = TRUE, verbose = TRUE)
    )
    
    # 5. View tuning results
    collect_metrics(rf_res)
    show_best(rf_res, metric = "accuracy", n = 5)
    autoplot(rf_res, metric = "accuracy")
    
    # 6. Select best model by accuracy
    best_rf <- select_best(rf_res, metric = "accuracy")
    
    # 7. Finalize and fit the model
    rf_final_wf <- finalize_workflow(rf_wf, best_rf)
    rf_fit <- fit(rf_final_wf, train)
    
    # 8. Make predictions on the test set
    rf_preds <- predict(rf_fit, test, type = "prob") %>%
        bind_cols(predict(rf_fit, test, type = "class")) %>%
        bind_cols(test %>% select(type))
    
})  # End runtime tracking

rf_time
# This prints out runtime (in seconds)

# 9. Confusion Matrix (basic)
cm_rf <- conf_mat(rf_preds, truth = type, estimate = .pred_class)
autoplot(cm_rf, type = "heatmap") +
    labs(title = "Random Forest — Confusion Matrix")

# 10. Confusion Matrix
cm_rf_df <- as.data.frame(cm_rf$table)
colnames(cm_rf_df) <- c("Truth","Prediction","Count")

cm_rf_df$Label <- NA
cm_rf_df$Label[cm_rf_df$Truth == "spam"    & cm_rf_df$Prediction == "spam"]    <- "True Positive"
cm_rf_df$Label[cm_rf_df$Truth == "nonspam" & cm_rf_df$Prediction == "nonspam"] <- "True Negative"
cm_rf_df$Label[cm_rf_df$Truth == "nonspam" & cm_rf_df$Prediction == "spam"]    <- "False Positive"
cm_rf_df$Label[cm_rf_df$Truth == "spam"    & cm_rf_df$Prediction == "nonspam"] <- "False Negative"

cm_rf_df$Percent <- round(cm_rf_df$Count / sum(cm_rf_df$Count) * 100, 1)

ggplot(cm_rf_df, aes(Prediction, Truth, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = paste0(Label,"\n",Count,"\n(",Percent,"%)")),
              fontface="bold", size=4) +
    scale_fill_gradient(low="#F7FBFF", high="orange") +
    labs(
        title = "Random Forest — Confusion Matrix (Labeled)",
        x = "Predicted Class",
        y = "Actual Class",
        fill = "Count"
    ) +
    theme_minimal(base_size = 13)

# 11. Full metrics
rf_metrics_out <- rf_preds %>%
    full_metrics(truth = type, estimate = .pred_class, .pred_spam)

rf_metrics_out

# 12. ROC Curve
roc_rf <- pROC::roc(
    response  = rf_preds$type,
    predictor = rf_preds$.pred_spam
)

plot(roc_rf, col = "darkorange", main = "Random Forest — ROC Curve")
pROC::auc(roc_rf)
# 13. Feature Importance Plot
rf_fit %>%
    extract_fit_parsnip() %>%
    vip(num_features = 15) +
    labs(title = "Random Forest - Top 15 Feature Importances")
```

## 📊 7. Model Comparison

| Model         | Accuracy  | Recall    | Precision | F1        |
|:--------------|:----------|:----------|:----------|:----------|
| k-NN          | 0.909     | 0.934     | 0.917     | 0.925     |
| SVM (RBF)     | 0.938     | 0.969     | 0.931     | 0.950     |
| Random Forest | **0.955** | **0.978** | **0.949** | **0.964** |

**Answer (Q16–Q19):**\
SVM outperformed k-NN across metrics, but Random Forest achieved the best overall balance of accuracy, recall, and scalability.

## 📚 References

-   scikit-learn. (2018). *Support Vector Machines*. [scikit-learn.org](https://scikit-learn.org/stable/modules/svm.html)\
-   scikit-learn. (2019). *RBF SVM Parameters*. [scikit-learn.org](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html)\
-   Hopkins, M., Reeber, E., Forman, G., & Suermondt, J. (1999). *Spambase Dataset.* Hewlett-Packard Labs.

💡 **Created with ❤ by Cierra Christian, Megan Geer, and Gary Mullings** Disclaimer(We ran the lab differently, so our outputs may be different)
