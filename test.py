import torch

@torch.jit.script
def foo(y):
    x = torch.nn.functional.tanhshrink(y)
    return x


print(foo.graph)
print(foo(torch.randn(2)))
