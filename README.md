# Proiect Machine Learning - Analiza Vanzarilor Restaurant

## Ce face proiectul?

Aplicam algoritmi de Machine Learning pe date de vanzari dintr-un restaurant pentru a:
- Prezice daca un client va cumpara un sos
- Recomanda sosuri pe baza cosului de cumparaturi
- Ranka produse pentru upselling

## Autori
Elisa Mercas & Denis Munteanu

---

## Cum rulezi proiectul

### Pas 1: Instaleaza dependentele
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook
```

### Pas 2: Pune dataset-ul in folder
Fisierul `ap_dataset.csv` trebuie sa fie in `data/raw/`

### Pas 3: Porneste Jupyter
```bash
python -m jupyter notebook notebooks/
```

Se deschide browser-ul. Daca nu, copiaza link-ul din terminal.

### Pas 4: Ruleaza notebook-urile in ordine
1. Click pe `01_eda.ipynb`
2. Meniu -> Kernel -> Restart & Run All
3. Astepti sa termine (nu mai ai `[*]` nicaieri)
4. Salvezi: Ctrl+S
5. Revii la lista de fisiere si faci la fel cu `02`, `03`, `04`, `05`

Ordine:
```
01_eda.ipynb              -> Explorare date
02_lr_crazy_sauce.ipynb   -> Model LR #1
03_lr_multi_sauce.ipynb   -> Model LR #2  
04_ranking_upsell.ipynb   -> Ranking basic
05_ranking_ml.ipynb       -> Ranking cu ML
```

### Pas 5: Raportul LaTeX
Mergi pe Overleaf (https://www.overleaf.com), uploadezi `report/report.tex` si folderul `results/`, compilezi si descarci PDF.

---

## Structura proiectului

```
data/raw/             - Dataset-ul (ap_dataset.csv)
src/                  - Cod sursa
  data_loader.py      - Incarcare date
  preprocessing.py    - Feature engineering
  models/
    logistic_regression.py  - LR from scratch
    evaluation.py           - Metrici + ROC-AUC
    ranking.py              - Naive Bayes + k-NN from scratch
notebooks/            - Jupyter notebooks (5 fisiere)
results/              - Grafice generate automat
report/               - Raport LaTeX
```

---

## Algoritmi implementati from scratch

| Algoritm            | Ce face                                   |
| ------------------- | ----------------------------------------- |
| Logistic Regression | Clasificare binara cu Gradient Descent    |
| Naive Bayes         | Clasificare probabilistica pentru ranking |
| k-NN                | Clasificare bazata pe vecini              |

---

## Ce genereaza notebook-urile

- Grafice cu distributia produselor
- Confusion matrix
- ROC curves
- Feature importance
- Comparatii Hit@K intre algoritmi

Toate se salveaza automat in `results/`.

---

## Probleme comune

**"jupyter" nu e recunoscut:**
```bash
python -m jupyter notebook notebooks/
```

**Kernel does not exist:**
Inchide toate tab-urile, Ctrl+C de 2 ori in terminal, reporneste Jupyter.

**Nu apare nimic in browser:**
Copiaza link-ul din terminal (cel cu `localhost:8888`).
