import torch
import torchvision.ops as tv

input_tensor = torch.rand(1000, 4)

scores = torch.rand(1000, 59)
#print(input_tensor)
#print(scores[1])

def NMS_cal(input_tensor, scores):
    NMS_group = []
    for i in range(59):
        confidscore, _ = torch.max(scores, dim=1)
        catargmax = torch.argmax(scores, dim=1)
        threshold = 0.5
        prop = input_tensor[catargmax==i]
        confid = confidscore[catargmax==i]
        NMS = tv.nms(prop, confid, threshold)
        NMS = NMS.tolist()
        NMS_group.append(NMS)
    return NMS_group    
NMS_g = NMS_cal(input_tensor, scores)

for i in range()
