"""
params.py -- parameters of the hybrid neuron model (Eqs. 1-3 of the paper), shared by neuron_figures.py.

    dv/dt = v^2 - w^2 + I_ext + I_TC,         I_TC = w0^2 / (1 - a^2)
    dw/dt = eps * (a v - w + w0)
    if v >= v_th:  v <- c,  w <- d

w0 is not set here: it is the parameter under study (w0 < 0 bistable, w0 >= 0 IF-like) and is passed
to every function. The network code (MNIST/, N_MNIST/, ODR/, navigation/) uses the same values.
"""

v_th = 1.0   # spike threshold
eps = 0.5    # time scale of the slow variable w
a = 0.1      # dependence of w on v
c = 0.0      # reset of v after a spike
d = 1.0      # reset of w after a spike
