# Test Datasets for TensorMap Training Charts

This directory contains 8 carefully crafted datasets for testing the live training charts feature.

## 📊 Datasets Overview

### 1. **quick_test.csv** ⚡ (RECOMMENDED FOR TESTING)
- **Type**: Multi-class Classification
- **Size**: 200 rows, 11 columns
- **Target**: `target` (3 classes)
- **Training Time**: ~10-15 seconds (5 epochs)
- **Best For**: Quick testing of charts functionality

**Usage Example**:
```python
Target Field: target
Problem Type: Multi class classification
Epochs: 5
Batch Size: 16
Training Split: 0.8
```

---

### 2. **iris_classification.csv** 🌸 (CLASSIC)
- **Type**: Multi-class Classification
- **Size**: 150 rows, 5 columns
- **Target**: `species` (setosa, versicolor, virginica)
- **Features**: sepal length, sepal width, petal length, petal width
- **Training Time**: ~8 seconds (10 epochs)
- **Best For**: Classic ML testing, well-behaved data

**Usage Example**:
```python
Target Field: species
Problem Type: Multi class classification
Epochs: 10
Batch Size: 16
Training Split: 0.8
Optimizer: adam
Metric: accuracy
```

**Expected Results**:
- Training accuracy: ~95-98%
- Validation accuracy: ~93-97%
- Loss should decrease smoothly

---

### 3. **wine_classification.csv** 🍷
- **Type**: Multi-class Classification
- **Size**: 178 rows, 14 columns
- **Target**: `wine_class` (0, 1, 2)
- **Features**: alcohol, malic_acid, ash, alkalinity, magnesium, etc.
- **Training Time**: ~12 seconds (15 epochs)
- **Best For**: Testing with more features

**Usage Example**:
```python
Target Field: wine_class
Problem Type: Multi class classification
Epochs: 15
Batch Size: 16
Training Split: 0.75
```

---

### 4. **customer_churn.csv** 📱
- **Type**: Binary Classification
- **Size**: 1000 rows, 8 columns
- **Target**: `churned` (0 or 1)
- **Features**: age, tenure, charges, contract type, payment method, etc.
- **Training Time**: ~20 seconds (20 epochs)
- **Best For**: Real-world business problem simulation

**Usage Example**:
```python
Target Field: churned
Problem Type: Multi class classification
Epochs: 20
Batch Size: 32
Training Split: 0.8
```

**Expected Behavior**:
- Churn rate: ~25%
- Model should show clear improvement over epochs
- Good for testing validation metrics

---

### 5. **housing_prices.csv** 🏠
- **Type**: Regression
- **Size**: 800 rows, 9 columns
- **Target**: `price` (continuous)
- **Features**: square_feet, bedrooms, bathrooms, age, lot_size, etc.
- **Training Time**: ~25 seconds (25 epochs)
- **Best For**: Testing regression visualization

**Usage Example**:
```python
Target Field: price
Problem Type: Linear Regression
Epochs: 25
Batch Size: 32
Training Split: 0.8
Optimizer: adam
Metric: mse
```

**Expected Results**:
- MSE should decrease steadily
- Watch for overfitting after epoch 15-20
- Good test for ETA calculation

---

### 6. **employee_attrition.csv** 👔
- **Type**: Binary Classification
- **Size**: 1200 rows, 10 columns
- **Target**: `left_company` (0 or 1)
- **Features**: age, years at company, salary, satisfaction, work-life balance, etc.
- **Training Time**: ~30 seconds (30 epochs)
- **Best For**: Testing longer training sessions

**Usage Example**:
```python
Target Field: left_company
Problem Type: Multi class classification
Epochs: 30
Batch Size: 32
Training Split: 0.8
```

**Best Use Case**:
- Test progress bar updates
- Test ETA accuracy over time
- Test stop/cancel functionality

---

### 7. **credit_card_fraud.csv** 💳
- **Type**: Imbalanced Binary Classification
- **Size**: 5000 rows, 9 columns
- **Target**: `is_fraud` (0 or 1, highly imbalanced ~0.4% fraud)
- **Features**: transaction amount, time, location, merchant data
- **Training Time**: ~45 seconds (40 epochs)
- **Best For**: Testing with imbalanced data

**Usage Example**:
```python
Target Field: is_fraud
Problem Type: Multi class classification
Epochs: 40
Batch Size: 64
Training Split: 0.8
```

**Challenges**:
- Imbalanced classes
- Model may show high accuracy but poor fraud detection
- Good for testing model performance visualization

---

### 8. **stock_prices.csv** 📈
- **Type**: Time Series Regression
- **Size**: 981 rows, 8 columns
- **Target**: `close` price
- **Features**: open, high, low, volume, moving averages
- **Training Time**: ~35 seconds (30 epochs)
- **Best For**: Time series testing

**Usage Example**:
```python
Target Field: close
Problem Type: Linear Regression
Epochs: 30
Batch Size: 32
Training Split: 0.8
```

---

## 🚀 Quick Start Guide

### Step 1: Upload Dataset
1. Go to TensorMap Data Upload page
2. Upload one of the CSV files
3. Verify columns are detected correctly

### Step 2: Create Model
1. Go to Model Builder
2. Create a simple neural network:
   - Input layer matching your features
   - 1-2 hidden layers (64-128 neurons)
   - Output layer matching your target

### Step 3: Configure Training
1. Go to Training page
2. Select your model
3. Choose dataset file
4. Configure parameters (see examples above)
5. Click "Save Configuration"

### Step 4: Start Training
1. Click "Train" button
2. **Watch the charts appear!** 🎉
3. Monitor real-time updates
4. Test stop button if needed

---

## 🧪 Testing Scenarios

### Scenario 1: Quick Smoke Test (2 minutes)
```
Dataset: quick_test.csv
Target: target
Epochs: 5
Batch Size: 16
Goal: Verify charts render and update
```

### Scenario 2: Monitor Full Training (30 seconds)
```
Dataset: iris_classification.csv
Target: species
Epochs: 20
Batch Size: 16
Goal: Watch complete training cycle, verify completion
```

### Scenario 3: Test Cancellation (15 seconds)
```
Dataset: employee_attrition.csv
Target: left_company
Epochs: 50
Batch Size: 32
Action: Click Stop after 10 epochs
Goal: Verify cancellation works
```

### Scenario 4: Test Page Reload (20 seconds)
```
Dataset: housing_prices.csv
Target: price
Epochs: 25
Batch Size: 32
Action: Reload page during training
Goal: Verify charts repopulate
```

### Scenario 5: Test Regression (25 seconds)
```
Dataset: housing_prices.csv
Target: price
Problem Type: Linear Regression
Metric: mse
Goal: Verify regression metrics display correctly
```

---

## 📋 Expected Chart Behaviors

### Classification Datasets
- **Loss**: Should decrease smoothly from ~1.0 to ~0.1-0.3
- **Accuracy**: Should increase from ~33-50% to ~85-98%
- **Validation**: Should track training closely
- **Best Epoch**: Usually 70-90% through training

### Regression Datasets
- **Loss (MSE)**: Should decrease exponentially
- **No Accuracy**: Accuracy chart should not appear
- **Validation Loss**: Should follow training loss
- **Best Epoch**: Watch for overfitting divergence

---

## 🐛 Troubleshooting

### Charts Don't Update
- **Check**: Is training actually running?
- **Check**: Browser console for errors
- **Try**: Reload page to trigger fallback

### Training Too Slow
- **Use**: quick_test.csv or iris_classification.csv
- **Reduce**: Epochs to 5-10
- **Reduce**: Batch size to 16

### Training Too Fast
- **Use**: employee_attrition.csv or credit_card_fraud.csv
- **Increase**: Epochs to 50+
- **Reduce**: Batch size to 16

### Poor Accuracy
- **Check**: Target field is correct
- **Check**: Problem type matches dataset
- **Try**: Different optimizer (adam → sgd)
- **Try**: More epochs

---

## 📊 Performance Benchmarks

| Dataset | Rows | Epochs | Time | Accuracy |
|---------|------|--------|------|----------|
| quick_test | 200 | 5 | ~10s | ~70% |
| iris | 150 | 10 | ~8s | ~95% |
| wine | 178 | 15 | ~12s | ~90% |
| churn | 1000 | 20 | ~20s | ~75% |
| housing | 800 | 25 | ~25s | N/A (MSE) |
| attrition | 1200 | 30 | ~30s | ~70% |
| fraud | 5000 | 40 | ~45s | ~99% |
| stocks | 981 | 30 | ~35s | N/A (MSE) |

*Benchmarks on M1 Mac with default TensorFlow settings*

---

## 🎯 Recommended Testing Order

1. **quick_test.csv** (5 epochs) - Verify basic functionality
2. **iris_classification.csv** (10 epochs) - Verify accuracy tracking
3. **housing_prices.csv** (20 epochs) - Verify regression
4. **employee_attrition.csv** (30 epochs) - Test cancellation
5. **credit_card_fraud.csv** (40 epochs) - Test longer training

---

## 💡 Pro Tips

1. **Fast Iteration**: Use quick_test.csv during development
2. **Visual Testing**: Use iris for clean, smooth curves
3. **Stress Testing**: Use credit_card_fraud for longer runs
4. **Regression Testing**: Use housing_prices for MSE charts
5. **Real-world**: Use customer_churn for realistic scenarios

---

## 📝 Dataset Characteristics

### Well-behaved (Smooth Training)
- ✅ quick_test.csv
- ✅ iris_classification.csv
- ✅ wine_classification.csv

### Challenging (May Show Noise)
- ⚠️ credit_card_fraud.csv (imbalanced)
- ⚠️ stock_prices.csv (time series)
- ⚠️ housing_prices.csv (complex patterns)

### Good for Features
- 🎯 **ETA Testing**: employee_attrition.csv (longer)
- 🎯 **Cancellation**: Any dataset with 30+ epochs
- 🎯 **Best Epoch**: iris, wine, churn
- 🎯 **Regression**: housing_prices.csv

---

## 🎉 Have Fun Testing!

These datasets are designed to showcase the live training charts in the best light. Start with quick_test.csv for instant gratification, then explore the others!

**Happy Training! 🚀**
