# Advanced Probabilistic Time Series Forecasting with LSTM + Attention

## GitHub Repository
```
https://github.com/<YourUsername>/advanced-time-series-attention
```

## Overview
- Implements **probabilistic forecasting** using **LSTM + Attention**.
- Predicts **3 quantiles (P10, P50, P90)** for each future time step.
- Captures **uncertainty** in forecasts.

## Approach / Methodology
1. **Data Preparation**
   - 10 related synthetic series generated with phase shift, trend, noise.
   - Normalized with MinMaxScaler.
   - Sliding window sequences created.

2. **Model Architecture**
   - LSTM layer for temporal patterns.
   - Attention layer for important time steps.
   - Output layer predicts 3 quantiles per series.

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
   - Loss: Quantile Loss

5. **Evaluation**
   - CRPS approximation printed.
   - Plot: median prediction + P10–P90 uncertainty bands.

## Files
| File | Description |
|------|-------------|
| `model.py` | LSTM + Attention + Probabilistic Quantile Network |
| `train.py` | Training, evaluation, plotting |
| `requirements.txt` | Python dependencies |
| `README.md` | Project explanation & instructions |

## Instructions to Run
1. Clone repo:
```
git clone https://github.com/<YourUsername>/advanced-time-series-attention
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Run training and evaluation:
```
python train.py
```
- CRPS prints in console.
- Plot shows median + P10–P90 bands.
## How to Run
1. Install required libraries using requirements.txt
2. Execute train.py to train the model
3. Review results in results.txt

## Evaluation Metrics
- RMSE
- MAE
