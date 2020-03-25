import torch
import torch.nn.functional as F

in1 = torch.DoubleTensor([[1,2],[3,4],[5,6],[7,8],[0,2]])
in2 = torch.DoubleTensor([[1,2],[3,4],[5,6],[7,8],[0,2]])

print (in1)
print (in2)

in1 = F.normalize(in1)
in2 = F.normalize(in2)
distance = torch.mm(in1, in2.t())
print distance