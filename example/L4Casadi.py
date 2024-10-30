import l4casadi as l4c


# Construct L4CasADi Model from PyTorch Model
l4casadi_model = l4c.L4CasADi(
pyTorch_model,
device='cpu', # Device in ['cpu', 'gpu', 'mps']
name='l4casadi_f' # Unique name
)
