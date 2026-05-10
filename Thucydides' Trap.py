# TT.py
# Soumadeep Ghosh
# Kolkata, India

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


# ------------------------------------------------------------
# 1. Training data
# ------------------------------------------------------------

data = [
    (
        "Late 15th century",
        "Portugal",
        "Spain",
        "Global empire and trade",
        "No War",
    ),
    (
        "First half of 16th century",
        "France",
        "Hapsburgs",
        "Land power in western Europe",
        "War",
    ),
    (
        "16 and 17 centuries",
        "Hapsburgs",
        "Ottoman Empire",
        "Land power in central and eastern Europe, sea power in the Mediterranean",
        "War",
    ),
    (
        "First half of 17 century",
        "Hapsburgs",
        "Sweden",
        "Land and sea power in northern Europe",
        "War",
    ),
    (
        "Mid-to-late 17 century",
        "Dutch Republic",
        "England",
        "Global empire, sea power, and trade",
        "War",
    ),
    (
        "Late 17 to mid-18th centuries",
        "France",
        "Great Britain",
        "Global empire and European land power",
        "War",
    ),
    (
        "Late 18th and early 19th centuries",
        "United Kingdom",
        "France",
        "Land and sea power in Europe",
        "War",
    ),
    (
        "Mid-19th century",
        "France and United Kingdom",
        "Russia",
        "Global empire, influence in Central Asia and eastern Mediterranean",
        "War",
    ),
    (
        "Mid-19th century",
        "France",
        "Germany",
        "Land power in Europe",
        "War",
    ),
    (
        "Late 19 and early 20 centuries",
        "China and Russia",
        "Japan",
        "Land and sea power in East Asia",
        "War",
    ),
    (
        "Early 20th century",
        "United Kingdom",
        "United States",
        "Global economic dominance and naval supremacy in the Western Hemisphere",
        "No War",
    ),
    (
        "Early 20th century",
        "United Kingdom supported by France, Russia",
        "Germany",
        "Land power in Europe and global sea power",
        "War",
    ),
    (
        "Mid-20th century",
        "Soviet Union, France, UK",
        "Germany",
        "Land and sea power in Europe",
        "War",
    ),
    (
        "Mid-20th century",
        "United States",
        "Japan",
        "Sea power and influence in the Asia-Pacific region",
        "War",
    ),
    (
        "1940s-1980s",
        "United States",
        "Soviet Union",
        "Global power",
        "No War",
    ),
    (
        "1990s-present",
        "United Kingdom and France",
        "Germany",
        "Political influence in Europe",
        "No War",
    ),
]

columns = [
    "period",
    "established_power",
    "rising_power",
    "domain",
    "outcome",
]

df = pd.DataFrame(data, columns=columns)

X = df[["period", "established_power", "rising_power", "domain"]]
y = df["outcome"]


# ------------------------------------------------------------
# 2. Text-feature logistic-regression classifier
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("period_text", TfidfVectorizer(ngram_range=(1, 2)), "period"),
        ("established_text", TfidfVectorizer(ngram_range=(1, 2)), "established_power"),
        ("rising_text", TfidfVectorizer(ngram_range=(1, 2)), "rising_power"),
        ("domain_text", TfidfVectorizer(ngram_range=(1, 2)), "domain"),
    ]
)

classifier = Pipeline(
    steps=[
        ("features", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            ),
        ),
    ]
)

classifier.fit(X, y)


# ------------------------------------------------------------
# 3. Prediction helper
# ------------------------------------------------------------

def predict_case(period, established_power, rising_power, domain):
    """
    Predict whether a power-transition case is classified as 'War' or 'No War'.
    """

    case = pd.DataFrame(
        [
            {
                "period": period,
                "established_power": established_power,
                "rising_power": rising_power,
                "domain": domain,
            }
        ]
    )

    prediction = classifier.predict(case)[0]

    probabilities = dict(
        zip(
            classifier.classes_,
            classifier.predict_proba(case)[0],
        )
    )

    return prediction, probabilities


# ------------------------------------------------------------
# 4. Example test cases
# ------------------------------------------------------------

examples = [
    (
        "21st century",
        "United States",
        "China",
        "Global economic dominance, naval power, technology, and Asia-Pacific influence",
    ),
    (
        "21st century",
        "India",
        "China",
        "Asian rivalry",
    ),
    (
        "21st century",
        "NATO",
        "Russia",
        "Mutual distrust and European security",
    ),
    (
        "21st century",
        "United States",
        "Iran",
        "Ideological differences and regional influence",
    ),
]

for case in examples:
    prediction, probabilities = predict_case(*case)

    print("\nCase:")
    print(f"  Period:             {case[0]}")
    print(f"  Established power:  {case[1]}")
    print(f"  Rising power:       {case[2]}")
    print(f"  Domain:             {case[3]}")
    print(f"Prediction:           {prediction}")
    print(f"Probabilities:        {probabilities}")


# ------------------------------------------------------------
# 5. Optional diagnostic: cross-validation
# ------------------------------------------------------------

scores = cross_val_score(
    classifier,
    X,
    y,
    cv=4,
    scoring="accuracy",
)

print("\nCross-validation accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
