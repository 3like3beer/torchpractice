import torch
from torch.autograd import Variable

###############################################################
# Create a variable:
x = Variable(torch.ones(2, 2), requires_grad=True)
print(x)

###############################################################
# Do an operation of variable:
y = x + 2
print(y)

###############################################################
# ``y`` was created as a result of an operation, so it has a ``grad_fn``.
print(y.grad_fn)