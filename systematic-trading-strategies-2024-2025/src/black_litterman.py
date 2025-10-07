import numpy as np

def black_litterman(weights, tau=0.05, view=0.02):
    w_prior = np.array(weights)
    w_view = w_prior + tau * (view - np.mean(w_prior))
    return w_view / np.sum(np.abs(w_view))
