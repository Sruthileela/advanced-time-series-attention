import numpy as np
from model import model

X = np.random.rand(100,10,1)
y = np.random.rand(100,1)

model.fit(X, y, epochs=5, batch_size=16)
