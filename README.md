# Proiect Machine Learning - Analiza Vânzărilor Restaurant

## Ce face proiectul?

Aplicăm algoritmi de Machine Learning pe date de vânzări dintr-un restaurant pentru a:
- **Prezice** dacă un client va cumpăra un sos
- **Recomanda** sosuri pe baza coșului de cumpărături
- **Rancka** produse pentru upselling

## Autori
- Elisa Mercas & Denis Munteanu

---

## 🚀 Cum rulezi proiectul

### Pas 1: Instalează dependențele
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter notebook
```

### Pas 2: Pune dataset-ul în folder
Fișierul `ap_dataset.csv` trebuie să fie în `data/raw/`

### Pas 3: Pornește Jupyter
```bash
cd C:\Users\Elisa\Desktop\Practical-Assignment-Machine-Learning-2025-Fall
python -m jupyter notebook notebooks/
```

Se deschide browser-ul. Dacă nu, copiază link-ul din terminal.

### Pas 4: Rulează notebook-urile în ordine
1. Click pe `01_eda.ipynb`
2. Meniu → **Kernel** → **Restart & Run All**
3. Aștepți să termine (nu mai ai `[*]` nicăieri)
4. Salvezi: Ctrl+S
5. Revii la lista de fișiere și faci la fel cu `02`, `03`, `04`, `05`

**Ordine:**
```
01_eda.ipynb              → Explorare date
02_lr_crazy_sauce.ipynb   → Model LR #1
03_lr_multi_sauce.ipynb   → Model LR #2  
04_ranking_upsell.ipynb   → Ranking basic
05_ranking_ml.ipynb       → Ranking cu ML
```

### Pas 5: Raportul LaTeX
Mergi pe [Overleaf](https://www.overleaf.com), creează cont, uploadezi `report/report.tex`, compilezi și descarci PDF.

---

## 📁 Structura proiectului

```
├── data/raw/             ← Dataset-ul (ap_dataset.csv)
├── src/                  ← Cod sursă
│   ├── data_loader.py    ← Încărcare date
│   ├── preprocessing.py  ← Feature engineering
│   └── models/
│       ├── logistic_regression.py  ← LR from scratch
│       ├── evaluation.py           ← Metrici + ROC-AUC
│       └── ranking.py              ← Naive Bayes + k-NN from scratch
├── notebooks/            ← Jupyter notebooks (5 fișiere)
├── results/              ← Grafice generate automat
└── report/               ← Raport LaTeX
```

---

## 🔧 Algoritmi implementați "from scratch"

| Algoritm                | Ce face                                   |
| ----------------------- | ----------------------------------------- |
| **Logistic Regression** | Clasificare binară cu Gradient Descent    |
| **Naive Bayes**         | Clasificare probabilistică pentru ranking |
| **k-NN**                | Clasificare bazată pe vecini              |

---

## 📊 Ce generează notebook-urile

- Grafice cu distribuția produselor
- Confusion matrix
- ROC curves
- Feature importance
- Comparații Hit@K între algoritmi

Toate se salvează automat în `results/`.

---

## ❓ Probleme comune

**"jupyter" nu e recunoscut:**
```bash
python -m jupyter notebook notebooks/
```

**Kernel does not exist:**
Închide toate tab-urile, Ctrl+C de 2 ori în terminal, repornește Jupyter.

**Nu apare nimic în browser:**
Copiază link-ul din terminal (cel cu `localhost:8888`).
