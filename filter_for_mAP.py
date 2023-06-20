import torch
import torchvision.ops as tv

input_tensor = torch.rand(1000, 4)

scores = torch.rand(1000, 60)
#print(input_tensor)
#print(scores[1])

def NMS_cal(input_tensor, scores):
    confidscore, _ = torch.max(scores, dim=1)
    catargmax = torch.argmax(scores, dim=1)
    threshold = 0.5
    
    combined_tensor = torch.cat((input_tensor, confidscore.unsqueeze(1), catargmax.unsqueeze(1)), dim=1)
    sorted_indices = torch.argsort(combined_tensor[:, 4:6], dim=1, descending=True)
    sorted_tensor = torch.gather(combined_tensor, 1, 
                                 (sorted_indices + 4).unsqueeze(2).expand(-1, -1, combined_tensor.shape[2]))
    print(sorted_tensor.size())
    #all_cb = sorted(all_cb,key=lambda x: x[-1])
    
    '''
    
    i = 0
    for c in (catargmax.size(0)):
        if c == catargmax[i].item:
            
            NMS = tv.nms(input_tensor, confidscore, threshold)
            
        print(NMS)
    '''
NMS_cal(input_tensor, scores)       