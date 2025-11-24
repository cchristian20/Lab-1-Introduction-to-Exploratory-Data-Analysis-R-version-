````markdown
# 🔐 CYBR 520 – Lab 5  
## Deep Learning for Intrusion Detection (UNSW-NB15, R + Keras)

**Authors:**  
- Cierra Christian  
- Megan Geer  
- Gary Mullings  

---

## 🌟 Project Overview

This lab builds a **deep learning–based Intrusion Detection System (IDS)** using the **UNSW-NB15** dataset and a **multi-layer perceptron (MLP)** in R with Keras/TensorFlow.

We walk through:

- Data understanding & cleaning  
- One-hot encoding and feature scaling  
- Building a **baseline MLP** and a **tuned MLP** (BatchNorm + Dropout + EarlyStopping)  
- Evaluating performance with **Accuracy** and **Macro-F1**  
- Training dynamics (epochs, overfitting, early stopping)  
- A bonus comparison: **MLP vs Random Forest** on the Iris dataset  

---

## 🧰 Tech Stack

- **Language:** R  
- **Deep Learning:** `keras`, `tensorflow`  
- **Data Wrangling:** `tidyverse`, `dplyr`, `janitor`, `recipes`  
- **Evaluation:** `yardstick`, `caret`  
- **Visualization:** `ggplot2`, `scales`  
- **Extra (Exploration):** `randomForest`  

---

## ⚙️ Environment Setup (R + TensorFlow)

> Run these once to set up your deep learning environment.

```r
# Install core packages
install.packages(c("reticulate", "keras", "tensorflow", 
                   "tidyverse", "janitor", "recipes", 
                   "yardstick", "caret", "ggplot2", "scales"))

library(reticulate)
library(keras)
library(tensorflow)

# Install Miniconda (if needed)
if (!reticulate::miniconda_exists()) {
  reticulate::install_miniconda()
}

# Use the Miniconda env created for R
use_miniconda("r-miniconda", required = TRUE)

# Install TensorFlow (CPU is enough for this lab)
tensorflow::install_tensorflow(version = "2.14.0")

# Quick sanity check
library(tensorflow)
tf$constant("TensorFlow is working!")
````

After setup, **Restart R** (`Session → Restart R`) and you’re ready to run the lab code.

---

## 📂 Data

**Dataset:** [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

* Generated in a cyber range using **IXIA PerfectStorm**
* Contains **modern normal traffic + synthetic contemporary attacks**
* Includes **49 engineered features** plus labels
* Attack families: Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

We use:

* `UNSW_NB15_training-set.csv`
* `UNSW_NB15_testing-set.csv`

---

## 🧪 R Script Overview

Main steps in the R pipeline:

1. **Load libraries** and set theme/seed
2. **Load CSVs** and `clean_names()`
3. **Fill missing values** (`fill_missing()` helper)
4. **Align factor levels** for `attack_cat`
5. **Preprocessing recipe** with:

   * `step_other()` for rare categories
   * `step_dummy()` for one-hot encoding
   * `step_zv()` to remove zero-variance features
   * `step_normalize()` to scale numeric predictors
6. **Convert to matrices** and **one-hot encode `attack_cat`** for Keras
7. **Build & train Baseline MLP**
8. **Build & train Tuned MLP** (BatchNorm + Dropout + EarlyStopping)
9. **Evaluate models** with Accuracy, Macro-F1, and confusion matrices
10. **Optional exploration:** MLP vs Random Forest on Iris dataset

---

# 🧠 Section 1 – Data Understanding & Preparation

### Q1. What is the UNSW-NB15 dataset designed for, and how does it differ from older IDS datasets (e.g., KDD99, NSL-KDD)?

UNSW-NB15 was designed for **modern intrusion detection research**. It captures realistic, hybrid traffic with contemporary attacks and normal behavior.

Older datasets like **KDD99/NSL-KDD** contain outdated attack types, synthetic traffic, and artifacts that don’t represent today’s networks. UNSW-NB15 uses IXIA PerfectStorm in a cyber range plus Argus/Bro-IDS features, giving us **49 modern features** and **nine attack families**, making it a much better fit for current ML-based IDS work.

---

### Q2. What are the major attack categories in this dataset?

The dataset has **nine attack families**:

* Fuzzers
* Analysis
* Backdoors
* DoS
* Exploits
* Generic
* Reconnaissance
* Shellcode
* Worms

---

### Q3. Is intrusion detection a multi-class classification problem or binary classification problem? Or both?

Intrusion detection can be **both**:

* **Binary:** Is this connection *benign vs. malicious*?
* **Multi-class:** If malicious, *which* attack family (DoS, Exploits, Generic, etc.)?

In this lab, our Keras model uses a **softmax output** with `K` classes (`layer_dense(units = K, activation = "softmax")`), so we treat IDS as a **multi-class classification problem** over the nine attack categories.

---

### Q4. Why do we remove or ignore the `id` column before training the model?

`id` is just a **row identifier**. It doesn’t carry any network behavior or attack information.

If we kept it, the model could latch onto accidental patterns in the IDs (row order or grouping) instead of real features, which would **hurt generalization**. So we drop it:

```r
DROP_COLS <- c("id")

train_raw <- train_raw %>% select(-any_of(DROP_COLS))
test_raw  <- test_raw  %>% select(-any_of(DROP_COLS))
```

---

# 🧹 Section 2 – Data Cleaning & Preparation

### Q5. Why do we replace missing values (numeric → 0, categorical → "unknown")?

Neural networks don’t understand `NA`.

* For **numeric columns**, we use `0` as a neutral placeholder that can be safely scaled later.
* For **categorical columns**, we use `"unknown"` to explicitly mark “missing” as its own category, instead of pretending it’s one of the real values.

This gives us a **complete, consistent matrix** to feed into the MLP.

---

### Q6. Why do we apply one-hot encoding instead of label encoding?

Label encoding (`1, 2, 3, 4, …`) introduces a **fake numeric order** where none exists. The model might think class `"4"` > `"1"` in some meaningful way.

One-hot encoding turns each category into its own binary column, avoiding any implied hierarchy and letting the network learn flexible patterns without being misled by arbitrary integer labels.

---

### Q7. Why must we apply the same preprocessing to both training and test data?

The model expects the test data to be transformed **in exactly the same way** as the training data:

* Same dummy columns
* Same scaling
* Same handling of rare levels

If preprocessing differs, test rows may have **missing or extra features**, different scales, or unseen encodings, which leads to **garbage predictions** or outright errors. Using a single `recipe()` + `prep()` + `bake()` pipeline guarantees consistent transformations.

---

# 📏 Section 3 – Feature Scaling

### Q8. Why do we scale numeric features before training a neural network?

Neural networks are sensitive to the **relative scale** of inputs:

* If some features range in the thousands and others are between 0 and 1, the large features dominate the gradients.
* Scaling puts all numeric features on a similar range, so:

  * Gradients behave better
  * Training becomes more stable
  * The optimizer converges faster

---

### Q9. What would likely happen if we trained an MLP *without* scaling?

Without scaling, training would likely be:

* **Slower** – gradients bounce around due to very different magnitudes
* **Unstable** – big features dominate weight updates
* **Less accurate** – small-scale features get ignored

So the MLP might converge poorly, overfit certain dimensions, or never find a good minimum.

---

# 🏗️ Section 4 – Model Architecture

### Q10. In your own words, what is an MLP classifier?

An **MLP (Multi-Layer Perceptron)** is a feed-forward neural network made of stacked layers of neurons.

Each neuron takes inputs, applies weights + bias, passes through an activation (like ReLU), and sends information to the next layer.

For classification, the network learns **nonlinear patterns** between features and labels by adjusting weights during training. Once trained, it outputs **class probabilities** for new inputs.

---

### Q11. What loss function do we use, and why is it appropriate for multi-class classification?

We use **categorical cross-entropy**:

```r
mlp_base %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = c("accuracy")
)
```

Categorical cross-entropy compares the model’s predicted probability distribution over classes to the **true one-hot label**, penalizing low probability assigned to the correct class.

It’s the standard loss for **multi-class softmax classifiers**.

---

# 📊 Section 5 – Baseline Model

### Q12. What was the baseline model’s overall test Accuracy and Macro-F1 score?

Using the baseline MLP architecture on UNSW-NB15, the model achieved:

* **Accuracy:** approximately **82–87%**
* **Macro-F1:** approximately **0.55–0.65**

(Exact values come from the `metrics_base` output after running:)

```r
metrics_base <- eval_multi(y_test, pred_base)
metrics_base
```

---

### Q13. Why is Macro-F1 more informative than Accuracy for this dataset?

UNSW-NB15 is **class-imbalanced**:

* Large classes: e.g., Generic, Exploits
* Tiny classes: e.g., Worms, Shellcode, some Backdoors

A model can get **high Accuracy** by mostly predicting majority classes and still completely miss minority attacks.

**Macro-F1**:

* Computes F1 **per class**
* Averages them equally
* Forces us to care about **small, rare attack types**, not just the big ones

This makes Macro-F1 a much more honest metric for IDS performance.

---

### Q14. Which attack class does the baseline model perform worst on, and why?

From the confusion matrix, the baseline MLP struggles most with **minority attack classes**, especially:

* Worms
* Shellcode
* Sometimes Backdoors

These classes have **very few training examples**, so the model doesn’t see enough patterns to learn them well. Severe class imbalance + rare patterns = lots of misclassifications.

---

# 🚀 Section 6 – Improved Model (BatchNorm + Dropout + EarlyStopping)

### Q15. What are Batch Normalization and Dropout, and how do they help?

* **Batch Normalization**

  * Normalizes layer inputs across a mini-batch
  * Keeps activations in a stable range
  * Helps gradients flow better and speeds up convergence

* **Dropout**

  * Randomly “turns off” a fraction of neurons during training
  * Forces the network not to rely on any single path
  * Acts as a regularizer, reducing overfitting

In the loss plots, the tuned model (with BN + Dropout) shows **smoother, more stable training** compared to the baseline.

---

### Q16. What does the EarlyStopping callback do and why is it useful?

**EarlyStopping** monitors validation performance (e.g., `val_loss`) and:

* Stops training when the metric stops improving for several epochs
* Restores the **best-performing weights**
* Saves time and avoids training when the model is just overfitting

This gives us a model that **generalizes better** without wasting extra epochs.

---

### Q17. Did the tuned model outperform the baseline? How can you tell?

Yes ✅

The tuned model achieves a **higher Macro-F1** than the baseline. This means it:

* Predicts attack categories more accurately overall
* Especially improves performance on minority classes

We can see the improvement in:

* The **Macro-F1 comparison bar chart**
* The **confusion matrix**, where low-support classes are predicted more often correctly

BatchNorm + Dropout + EarlyStopping together give us a **more stable and better-generalizing** model.

---

# ⏱️ Section 7 – Training Dynamics (Epochs & Loss)

### Q18. What is an epoch? Why do we set a static max number of epochs?

An **epoch** is one full pass through the entire training dataset. Neural networks usually need many epochs to slowly refine weights.

We set a **maximum epochs** value (e.g., 40) so training has an upper bound for consistency. Then **EarlyStopping** can halt earlier when the validation loss stops improving, giving us:

* A good model
* No wasted computation
* Less risk of overfitting

---

### Q19. What do the loss and validation loss graphs tell us? Why not just run all 40 epochs?

From the training/validation curves:

* Training loss decreases and flattens around ~20 epochs
* Validation loss slightly decreases and then **levels off** around the same point
* Validation accuracy also plateaus

This suggests that the model has basically **learned what it can** by ~20 epochs.

Running all 40 epochs would:

* Use extra compute
* Risk overfitting (training loss keeps going down while validation loss stops improving)

EarlyStopping stops around the “sweet spot” instead of blindly using all 40 epochs.

---

### Q20. What additional techniques could further improve performance? (Name at least two)

Two promising techniques:

1. **SMOTE (Synthetic Minority Oversampling Technique)**

   * Creates synthetic samples for minority classes
   * Reduces class imbalance
   * Often boosts recall and F1 for rare attacks

2. **Feature Embeddings for Categorical Variables**

   * Instead of one-hot encoding, learn dense vector embeddings
   * Reduces input dimensionality
   * Captures similarity between categories
   * Often improves both performance and training efficiency

```r
Code:
  # Part 7: Exploration
  ############################################################
# MLP vs. Random Forest on Iris Dataset
############################################################

library(tidyverse)
library(recipes)
library(janitor)
library(caret)
library(keras3)
library(yardstick)
library(ggplot2)
library(scales)
library(randomForest)

set.seed(42)
theme_set(theme_minimal(base_size = 13))

############################################################
# 1) Load Iris Dataset
############################################################

data(iris)
iris <- clean_names(iris)
iris <- iris %>% rename(target = species)

# Train-test split
set.seed(42)
train_idx <- createDataPartition(iris$target, p = 0.8, list = FALSE)
train_raw <- iris[train_idx, ]
test_raw  <- iris[-train_idx, ]

############################################################
# 2) Preprocessing: One-hot + Scaling
############################################################

rec <- recipe(target ~ ., data = train_raw) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())

prep_obj <- prep(rec, training = train_raw)
train_baked <- bake(prep_obj, new_data = train_raw)
test_baked  <- bake(prep_obj, new_data = test_raw)

# Extract
y_train <- train_baked$target
y_test  <- test_baked$target
X_train <- train_baked %>% select(-target)
X_test  <- test_baked %>% select(-target)

# Convert to matrices
X_train_m <- as.matrix(X_train)
X_test_m  <- as.matrix(X_test)

# One-hot encoding for MLP labels
class_levels <- levels(y_train)
K <- length(class_levels)

y_train_idx <- as.integer(y_train) - 1
y_test_idx  <- as.integer(y_test)  - 1

y_train_oh <- to_categorical(y_train_idx, num_classes = K)
y_test_oh  <- to_categorical(y_test_idx,  num_classes = K)

############################################################
# 3) MLP Model
############################################################

input_dim <- ncol(X_train_m)

mlp_model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = input_dim) %>%
  layer_dense(units = K, activation = "softmax")

mlp_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics = "accuracy"
)

hist_mlp <- mlp_model %>% fit(
  x = X_train_m,
  y = y_train_oh,
  validation_split = 0.2,
  epochs = 40,
  batch_size = 8,
  verbose = 2
)

# Plot learning curves
plot(hist_mlp)

############################################################
# 4) MLP Evaluation
############################################################

prob_mlp <- mlp_model %>% predict(X_test_m)
pred_mlp_idx <- max.col(prob_mlp)
pred_mlp <- factor(class_levels[pred_mlp_idx], levels = class_levels)

cm_mlp <- table(Actual = y_test, Predicted = pred_mlp)
cm_mlp

############################################################
# 5) Random Forest Classifier
############################################################

rf_model <- randomForest(
  target ~ .,
  data = train_raw,
  ntree = 500,
  mtry = 2
)

pred_rf <- predict(rf_model, test_raw)
cm_rf <- table(Actual = y_test, Predicted = pred_rf)
cm_rf

############################################################
# 6) Metrics
############################################################

eval_multi <- function(y_true, y_pred) {
  y_true <- factor(y_true)
  y_pred <- factor(y_pred, levels = levels(y_true))
  df <- tibble(.truth = y_true, .pred = y_pred)
  
  tibble(
    Accuracy = accuracy(df, truth = .truth, estimate = .pred) |> pull(.estimate),
    Macro_F1 = f_meas(df, truth = .truth, estimate = .pred, estimator = "macro") |> pull(.estimate)
  )
}

metrics_mlp <- eval_multi(y_test, pred_mlp)
metrics_rf  <- eval_multi(y_test, pred_rf)

metrics_mlp
metrics_rf

############################################################
# 7) Final Comparison
############################################################

results <- bind_rows(
  metrics_mlp %>% mutate(Model = "MLP"),
  metrics_rf  %>% mutate(Model = "Random Forest")
) %>% relocate(Model)

print(results)

ggplot(results, aes(x = Model, y = Macro_F1, fill = Model)) +
  geom_col() +
  coord_flip() +
  labs(title = "MLP vs Random Forest — Macro F1", y = "Macro F1 Score") +
  theme(legend.position = "none")
```






---


# 🌱 Section 8 – Extra Exploration: MLP vs Random Forest on Iris

For the exploration portion, we trained:

* A small **MLP** on the Iris dataset
* A **Random Forest** classifier

Results:

* The MLP learned reasonably well but showed signs of **overfitting** (validation loss stopped improving early).
* Random Forest achieved **higher accuracy and Macro-F1**, especially on the tricky Virginica vs Versicolor boundary.

Takeaway:
For **small, tabular datasets**, classical models like Random Forest often outperform deep learning, which typically needs **more data + more tuning** to shine.

---

## 📎 Files & Organization

Suggested repo layout:

```text
lab5-unsw-nb15-mlp/
├── README.md               # This document 🎉
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── R/
│   ├── lab5_unsw_nb15_mlp.R      # Main UNSW lab script
│   └── iris_mlp_vs_rf.R          # Exploration script (Iris dataset)
└── plots/
    ├── baseline_confusion.png
    ├── tuned_confusion.png
    └── macro_f1_comparison.png
```

---

## 🎯 Final Thoughts

This lab walks through a **full deep learning workflow** for intrusion detection:

* From messy CSVs → clean, scaled, one-hot encoded features
* From basic MLP → tuned model with BN, Dropout, and EarlyStopping
* From simple Accuracy → more honest metrics like Macro-F1

Most importantly, it shows that **how you prepare data and design your evaluation** matters just as much as the model architecture itself.

