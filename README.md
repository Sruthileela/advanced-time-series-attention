# Advanced Probabilistic Time Series Forecasting with LSTM + Attention

## 📌 Project Overview
This project implements **probabilistic forecasting** using **LSTM + Attention**.  
Instead of predicting a single value, the model predicts three quantiles (P10, P50, P90), giving a **range of possible future values**.

---

## 🧠 Approach / Methodology
1. **Data Preparation**  
   - 10 related synthetic time series generated  
   - Normalized using MinMaxScaler  
   - Converted to sliding window sequences

2. **Model Architecture**  
   - LSTM Layer: Captures temporal patterns  
   - Attention Layer: Focuses on important time steps  
   - Output Layer: Predicts multiple quantiles

3. **Quantile Loss**  
```python
def quantile_loss(preds, target, quantiles=[0.1,0.5,0.9]):
    total_loss = 0
    for i, q in enumerate(quantiles):
        e = target - preds[:,:,:,i]
        total_loss += torch.mean(torch.max(q*e, (q-1)*e))
    return total_loss
```

4. **Training**  
   - Optimizer: Adam  
   - Epochs: 50  
   - Batch size: 64  

5. **Evaluation**  
   - CRPS approximation  
   - Visual: Actual vs median prediction + P10–P90 band

---

## 📁 Files
| File | Description |
|------|-------------|
| `model.py` | LSTM + Attention + Probabilistic Quantile Network |
| `train.py` | Training, evaluation, plotting |
| `requirements.txt` | Required packages |
| `README.md` | Project explanation & instructions |

---

## 🚀 Instructions to Run
1. Clone repo:  
```
git clone https://github.com/<YourUsername>/advanced-time-series-attention
```
2. Install dependencies:  
```
pip install -r requirements.txt
```
3. Run training & evaluation:  
```
python train.py
```

---

## 🔎 Interpretation
- Outputs 3 quantiles: P10 (lower), P50 (median), P90 (upper)  
- Probabilistic forecasting shows **uncertainty bounds**  
- CRPS approx printed in console, plots show prediction bands

---

## 📌 GitHub Repository
```
https://github.com/<YourUsername>/advanced-time-series-attention
```
