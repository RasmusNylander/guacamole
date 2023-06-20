import torch

input_tensor = torch.rand(1000, 4)

scores = torch.rand(1000, 60)
#print(input_tensor)
#print(scores[1])

confidscore, _ = torch.max(scores, dim=1)
catargmax = torch.argmax(scores, dim=1)

for c in range(60):
    
