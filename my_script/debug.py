import torch
import torch.nn.functional as F

# a = torch.randint(2,size=(3,2),dtype=torch.int)
# print(a)
# a = a.bool()
# print("-----a-------")
# print(a)
# b = a.logical_not()
# print("-----b-------")
# print(b)
# c = torch.zeros_like(b, dtype=torch.float)
# print("-----c-------")
# print(c)
# c.masked_fill_(b, float("-inf"))
# print("-----c-------")
# print(c)

# a = torch.randint(2, size=(1280,1280), dtype=torch.int).bool()
# b = a.logical_not()
# c = torch.zeros_like(a, dtype=torch.float)
# c.masked_fill_(b, float("-inf"))
# print(c.shape)
# mask = F.interpolate(c[None, None], scale_factor=1/16, mode='nearest')
# print(mask.shape)
# mask = mask.reshape([1, -1, 1])
# print(mask.shape)

a = torch.from_numpy([[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]).bool()
print(a)
b = a.logic
