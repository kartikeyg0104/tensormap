# 🚀 Quick Reference - Test Datasets

## Fastest Testing (Choose One)

### ⚡ Ultra Fast (10 seconds)
```
File: quick_test.csv
Target: target
Type: Multi class classification
Epochs: 5
Batch: 16
Split: 0.8
```

### 🌸 Classic (8 seconds)
```
File: iris_classification.csv
Target: species
Type: Multi class classification
Epochs: 10
Batch: 16
Split: 0.8
```

### 🏠 Regression (25 seconds)
```
File: housing_prices.csv
Target: price
Type: Linear Regression
Epochs: 25
Batch: 32
Split: 0.8
Metric: mse
```

---

## All Datasets at a Glance

| Dataset | Type | Rows | Time | Target | Best For |
|---------|------|------|------|--------|----------|
| quick_test.csv | Class | 200 | 10s | target | ⚡ Speed |
| iris_classification.csv | Class | 150 | 8s | species | 🎯 Accuracy |
| wine_classification.csv | Class | 178 | 12s | wine_class | 📊 Features |
| customer_churn.csv | Binary | 1000 | 20s | churned | 💼 Business |
| housing_prices.csv | Regr | 800 | 25s | price | 🏠 Regression |
| employee_attrition.csv | Binary | 1200 | 30s | left_company | ⏱️ Duration |
| credit_card_fraud.csv | Binary | 5000 | 45s | is_fraud | 🔬 Challenge |
| stock_prices.csv | Regr | 981 | 35s | close | 📈 Time Series |

---

## Copy-Paste Configs

### For Classification
```
Problem Type: Multi class classification
Optimizer: adam
Metric: accuracy
Training Split: 0.8
```

### For Regression
```
Problem Type: Linear Regression
Optimizer: adam
Metric: mse
Training Split: 0.8
```

---

## Common Settings

**Fast Training (demo)**
- Epochs: 5-10
- Batch Size: 16
- Dataset: quick_test.csv or iris

**Normal Training (testing)**
- Epochs: 20-30
- Batch Size: 32
- Dataset: churn or housing

**Long Training (stress test)**
- Epochs: 40-50
- Batch Size: 32
- Dataset: fraud or attrition

---

## Testing Checklist

- [ ] Start training → Charts appear
- [ ] Watch epochs update → Lines extend
- [ ] Check progress bar → Updates correctly
- [ ] Check ETA → Reasonable estimate
- [ ] Click stop button → Training cancels
- [ ] Reload page → Charts repopulate
- [ ] Complete training → Green badge shows

---

## Troubleshooting

**No charts?**
→ Check browser console (F12)

**Too slow?**
→ Use quick_test.csv with 5 epochs

**Too fast?**
→ Use fraud.csv with 40 epochs

**Bad accuracy?**
→ Check target field name

**Need regression?**
→ Use housing_prices.csv

---

📁 **Location**: `tensormap-backend/test_datasets/`

📖 **Full Guide**: See `README.md` in this folder

🎯 **Start Here**: Use `quick_test.csv` for first test!
