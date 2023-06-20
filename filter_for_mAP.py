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
    prop = input_tensor[catargmax==1]
    confid = confidscore[catargmax==1]
    NMS = tv.nms(prop, confid, threshold)
    print(NMS)
    
    '''
    combined_tensor = torch.cat((input_tensor, confidscore.unsqueeze(1), catargmax.unsqueeze(1)), dim=1)
    sorted_indices = torch.argsort(combined_tensor[:, 4:6], dim=1, descending=True)
    sorted_tensor = torch.cat((combined_tensor[:, :4], combined_tensor[:, 4:6].gather(1, sorted_indices)), dim=1)
    print(combined_tensor[0])
    print(sorted_tensor[0])
    #all_cb = sorted(all_cb,key=lambda x: x[-1])
    
    
    
    i = 0
    for c in (catargmax.size(0)):
        if c == catargmax[i].item:
            
            NMS = tv.nms(input_tensor, confidscore, threshold)
            
        print(NMS)
    '''
NMS_cal(input_tensor, scores)       