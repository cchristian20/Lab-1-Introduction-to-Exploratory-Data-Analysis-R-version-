# CYBR 520 – Lab 1: Exploratory Data Analysis (R)
# Cierra Christian, Megan Geer, Samantha Easton, and Gary Mullings

This is a step-by-step guide that anyone can follow to reproduce the lab from a clean machine.  
You’ll (1) set up R/RStudio, (2) analyze the built-in **Iris** dataset, and (3) create plots and visualizations for your report.

---

## 🌸 Dataset Overview

**Dataset:** Iris  
**Topics:** Summary statistics, data visualization, variable relationships, feature distributions, and outlier detection  
**R Packages:** `tidyverse`, `tidymodels`, `GGally`  

### Background

The **Iris dataset** is a classic example used in R for machine learning and statistical analysis.  
Most, if not all, flowers have a **sepal** and a **petal**. The **sepal** (Figure 1) functions as a protector for the flower and supports the petals when it blooms.

![Figure 1 – Petal vs Sepal](images/c41f3ed9-f1b1-4bc8-9b44-a3fd24cda6aa.png)

The Iris dataset contains measurements (in cm) of **petal length and width**, and **sepal length and width** for three species of iris flowers:  
**Iris versicolor**, **Iris setosa**, and **Iris virginica** (Figure 2).  
Each species includes **50 observations**, for a total of **150 rows**.  
Each record also includes a class label identifying the species (“versicolor”, “setosa”, or “virginica”).

![Figure 2 – Iris Species](images/1c98d6d7-f6b4-4915-a639-80c186bc4a95.png)

---

## ✅ 0) Prerequisites

- **R** (version 4.x or newer): https://cran.r-project.org/  
- **RStudio Desktop**: https://posit.co/download/rstudio-desktop/ *(optional but recommended)*  
- Git + GitHub account (optional, for version control)

> 💡 If you’re brand new: install R first, then RStudio.

---

## 📁 1) Create Your Project Structure

Create a folder for your repo (locally or by cloning an empty GitHub repo). Inside it, make this layout:

```
your-repo/
├── R/                       # R scripts
│   └── lab1_eda.R          # main script (we'll create this)
├── images/                  # exported figures (auto-saved)
├── README.md                # this file
└── .gitignore               # optional
```

> We will write all analysis into `R/lab1_eda.R` and export plots to `images/`.

---

## 🧰 2) Install Required Packages (one-time)

Open R or RStudio Console and run:

```r
install.packages(c("tidyverse", "tidymodels", "GGally"))
```

---

## 🧪 3) Create the Analysis Script

Create a new file at `R/lab1_eda.R` and paste everything from the sections below **in order**.  
Then you can run chunks line-by-line or the whole file.

---

## ⚙️ 3.1 Load Libraries & Data

```r
library(tidyverse)
library(tidymodels)
library(GGally)

data("iris")                     # built-in dataset (150 rows × 5 columns)
iris_tbl <- as_tibble(iris)      # tibble format for cleaner printing
glimpse(iris_tbl)
summary(iris_tbl)
```

**Expected:** 150 rows × 5 columns.  
Numeric variables = `Sepal.Length`, `Sepal.Width`, `Petal.Length`, `Petal.Width`; Factor = `Species`.

---

## 📊 4) Descriptive Statistics (Range, SD, etc.)

### 4.1 Quick Min/Max/Range/SD for Each Numeric Feature

```r
desc_stats <- iris_tbl %>%
  summarise(across(where(is.numeric),
                   list(min = min, max = max, range = ~max(.) - min(.), sd = sd),
                   .names = "{.col}_{.fn}"))
desc_stats
```

### 4.2 Per-Species Mean/Median/SD (Grouped Summary)

```r
sum_by_species <- iris_tbl %>%
  group_by(Species) %>%
  summarise(across(where(is.numeric),
                   list(mean = mean, median = median, sd = sd),
                   .names = "{.col}_{.fn}"))
sum_by_species
```

---

## 📈 5) Univariate Visualizations

> These help visualize shape (skew/symmetry), spread, and potential outliers.

### 5.1 Histograms by Species

```r
p_hist_sepal_len <- ggplot(iris_tbl, aes(Sepal.Length, fill = Species)) +
  geom_histogram(alpha = 0.6, bins = 20) +
  labs(title = "Distribution of Sepal Length by Species")

p_hist_sepal_wid <- ggplot(iris_tbl, aes(Sepal.Width, fill = Species)) +
  geom_histogram(alpha = 0.6, bins = 20) +
  labs(title = "Distribution of Sepal Width by Species")

p_hist_petal_len <- ggplot(iris_tbl, aes(Petal.Length, fill = Species)) +
  geom_histogram(alpha = 0.6, bins = 20) +
  labs(title = "Distribution of Petal Length by Species")

p_hist_petal_wid <- ggplot(iris_tbl, aes(Petal.Width, fill = Species)) +
  geom_histogram(alpha = 0.6, bins = 20) +
  labs(title = "Distribution of Petal Width by Species")

p_hist_sepal_len; p_hist_sepal_wid; p_hist_petal_len; p_hist_petal_wid
```

---

### 5.2 Boxplots by Species (Spot Outliers)

```r
box_by_species <- function(var) {
  ggplot(iris_tbl, aes(x = Species, y = {{ var }}, fill = Species)) +
    geom_boxplot() +
    labs(title = paste("Boxplot of", rlang::as_name(ensym(var)), "by Species"))
}

p_box_sepal_len <- box_by_species(Sepal.Length)
p_box_sepal_wid <- box_by_species(Sepal.Width)
p_box_petal_len <- box_by_species(Petal.Length)
p_box_petal_wid <- box_by_species(Petal.Width)

p_box_sepal_len; p_box_sepal_wid; p_box_petal_len; p_box_petal_wid
```

> **Observation:** *Iris virginica* often shows greater variability in petal dimensions.

---

## 🔗 6) Bivariate Relationships (Pairs & Scatterplots)

### 6.1 Pairwise Scatterplot Matrix (`ggpairs`)

```r
p_pairs <- ggpairs(iris_tbl, aes(color = Species))
p_pairs
```

**Key observations:**
- **Petal.Length vs Petal.Width:** Strongest positive relationship (≈ 0.96)  
- **Sepal.Length vs Petal.Length:** Moderate positive  
- **Sepal.Width:** Generally weaker correlations  

---

### 6.2 Focused Scatterplots (Sepal vs Sepal, Petal vs Petal)

```r
p_sepal_plane <- ggplot(iris_tbl, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Sepal Dimensions Overlap Between Species",
       x = "Sepal Length (cm)", y = "Sepal Width (cm)")

p_petal_plane <- ggplot(iris_tbl, aes(Petal.Length, Petal.Width, color = Species)) +
  geom_point(size = 3, alpha = 0.7) +
  labs(title = "Petal Dimensions Clearly Separate Species",
       x = "Petal Length (cm)", y = "Petal Width (cm)")

p_sepal_plane; p_petal_plane
```

> **Interpretation:** Petal features show clear class separation, while sepal features overlap (especially *versicolor* vs *virginica*).

---

## 🔥 7) Correlations (Heatmap)

```r
cor_matrix <- cor(iris_tbl %>% select(where(is.numeric)))

cor_data <- as.data.frame(cor_matrix) %>%
  rownames_to_column("Var1") %>%
  pivot_longer(cols = -Var1, names_to = "Var2", values_to = "value")

p_heat <- ggplot(cor_data, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 2)), size = 4) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  labs(title = "Correlation Heatmap of Iris Numeric Variables", fill = "Correlation") +
  theme_minimal()

p_heat
```

> Confirms that **Petal.Length ↔ Petal.Width** have the strongest correlation (≈ 0.96).

---

## 💾 8) Export All Figures (PNG)

```r
if (!dir.exists("images")) dir.create("images")

ggsave("images/hist_sepal_length.png", p_hist_sepal_len, width = 8, height = 6, dpi = 300)
ggsave("images/hist_sepal_width.png",  p_hist_sepal_wid, width = 8, height = 6, dpi = 300)
ggsave("images/hist_petal_length.png", p_hist_petal_len, width = 8, height = 6, dpi = 300)
ggsave("images/hist_petal_width.png",  p_hist_petal_wid, width = 8, height = 6, dpi = 300)

ggsave("images/box_sepal_length.png",  p_box_sepal_len, width = 8, height = 6, dpi = 300)
ggsave("images/box_sepal_width.png",   p_box_sepal_wid, width = 8, height = 6, dpi = 300)
ggsave("images/box_petal_length.png",  p_box_petal_len, width = 8, height = 6, dpi = 300)
ggsave("images/box_petal_width.png",   p_box_petal_wid, width = 8, height = 6, dpi = 300)

ggsave("images/ggpairs_matrix.png",    p_pairs,         width = 10, height = 8, dpi = 300)
ggsave("images/sepal_scatter.png",     p_sepal_plane,   width = 8,  height = 6, dpi = 300)
ggsave("images/petal_scatter.png",     p_petal_plane,   width = 8,  height = 6, dpi = 300)
ggsave("images/cor_heatmap.png",       p_heat,          width = 8,  height = 6, dpi = 300)
```

---

## 🧪 9) Reproducibility Notes

- No random sampling used → same results on any R 4.x setup.  
- To check versions:
```r
sessionInfo()
```

---

## 🧯 10) Troubleshooting

- **Plots not saving?** Ensure `images/` exists and you have write permissions.  
- **Function not found?** Re-run `install.packages(...)` and `library(...)`.  
- **Encoding issues?** Replace special symbols like “×” with “x”.

---

## 📚 References

- Fisher, R. A. (1936). *The Use of Multiple Measurements in Taxonomic Problems.* *Annals of Eugenics.*  
- R Documentation: `?iris`  
- GGally Documentation: [https://ggobi.github.io/ggally/](https://ggobi.github.io/ggally/)

---

## 📦 What to Commit

```
your-repo/
├── R/
│   └── lab1_eda.R
├── images/
│   ├── ggpairs_matrix.png
│   ├── petal_scatter.png
│   ├── sepal_scatter.png
│   ├── cor_heatmap.png
│   ├── hist_sepal_length.png
│   ├── hist_sepal_width.png
│   ├── hist_petal_length.png
│   ├── hist_petal_width.png
│   ├── box_sepal_length.png
│   ├── box_sepal_width.png
│   ├── box_petal_length.png
│   └── box_petal_width.png
└── README.md
```

---

## 🏁 Quick Run (All at Once)

From the repo root in a terminal:

```bash
Rscript R/lab1_eda.R
```

Figures will appear under `images/`.  
Open them and embed in your README or report as needed.
