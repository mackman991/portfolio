# Customer Analytics & Predictive Modelling — Turtle Games

End-to-end customer analytics project for Turtle Games, a global game manufacturer
and retailer. Covers regression modelling, customer segmentation and NLP sentiment
analysis to identify loyalty drivers and inform marketing strategy.

LSE Data Analytics Career Accelerator — Course 3 assignment.

## Objectives

1. Identify the key drivers of customer loyalty points accumulation
2. Segment the customer base into actionable groups for targeted marketing
3. Extract insight from unstructured product review text

## Key results

### Regression modelling
- Built and compared simple linear, multiple linear and decision-tree regression models
  on 2,000 customer records
- **Decision tree regressor: R² = 0.96** predicting loyalty points from income and spending
  score — materially outperforming linear models, which exhibited heteroscedasticity
- Log-transforming the target (loyalty points) improved linear model fit; income and
  spending score are the dominant predictors; age adds marginal explanatory power

### Customer segmentation
- Applied k-means clustering with **elbow method + silhouette score** validation
- Identified **5 distinct customer segments** by income and spending score
- Segment 1 (high income / low spend): highest loyalty-point potential — recommended
  targeted incentive programmes to convert latent value
- PCA-reduced visualisation confirms clean cluster separation

### NLP sentiment analysis
- Applied **VADER sentiment analysis** to product reviews and summary fields
- Generated polarity and subjectivity scores across the corpus
- Word-cloud and frequency analysis surface dominant positive themes; identified a
  systematic positive-feedback bias in summary fields vs full reviews

## Files

| File | Description |
|---|---|
| `Mackin_Peter_DA301_Assignment_Notebook.ipynb` | Full Python analysis notebook |
| `Mackin_Peter_DA301_Assignment_Rscript.R` | R script (exploratory analysis) |
| `Mackin_Peter_DA301_Assignment_Report.pdf` | Written report submitted to LSE |

## Tools & methods

- **Python** — pandas, scikit-learn, statsmodels, NLTK (VADER), matplotlib, seaborn
- **R** — ggplot2, exploratory analysis
- Simple & multiple linear regression
- Decision tree regression (scikit-learn `DecisionTreeRegressor`)
- k-means clustering with PCA visualisation
- VADER sentiment scoring

## Setup

```bash
pip install pandas scikit-learn statsmodels nltk matplotlib seaborn jupyter
python -c "import nltk; nltk.download('vader_lexicon')"
jupyter notebook Mackin_Peter_DA301_Assignment_Notebook.ipynb
```
