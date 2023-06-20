import tqdm
import torch
from dataloader import Patches, DatasetType, ProposalsEval
from torch.utils.data import DataLoader
import os
from networks import Architecture
from assert_gpu import assert_gpu
import torchvision
from mAP import calc_AP
device = assert_gpu()

def nms(boxes, scores, iou_threshold):
    # Sort the scores in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    keep_indices = []

    while sorted_indices.numel() > 0:
        # Get the index with the highest score
        best_idx = sorted_indices[0]
        keep_indices.append(best_idx.item())

        # Calculate the IoU between the best box and the rest of the boxes
        best_box = boxes[best_idx]
        rest_boxes = boxes[sorted_indices[1:]]
        ious = calculate_iou(best_box, rest_boxes)

        # Find the indices of boxes that have IoU less than the threshold
        mask = ious <= iou_threshold
        sorted_indices = sorted_indices[1:][mask]

    # Return the indices of the kept elements in descending order of scores
    return torch.tensor(keep_indices, dtype=torch.int64)

def calculate_iou(box, boxes):
    # Calculate the intersection coordinates
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])

    # Calculate the intersection area
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Calculate the areas of the boxes
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate the union area
    union = box_area + boxes_area - intersection

    # Calculate the IoU
    iou = intersection / union

    return iou

def evaluate(model):

    proposal_set = ProposalsEval(root_dir= "data_wastedetection",ds_type = DatasetType.train)
    img_loader = DataLoader(proposal_set, batch_size=1, shuffle=False, num_workers=1)

    cat_scores = [torch.tensor([]) for _ in range(59)]
    cat_trues = [torch.tensor([]) for _ in range(59)]
    #for img_num, (proposals, image, true_bb, true_cat) in enumerate(tqdm(img_loader, leave=False, unit="batches", position=1)):
    for (proposals, image, true_bb, true_cat) in img_loader:

        image = image.squeeze()
        proposals = proposals.squeeze()
    
        batch_size = 64
        patch_set = Patches(image, proposals)
        patch_loader = DataLoader(patch_set, batch_size=batch_size, shuffle=False, num_workers=1)
    
        predictions = []
        for patch in patch_loader:
            patch.to(device).float()
            predictions.append(model(patch))

        predictions = torch.cat(predictions)

        confidscore, catargmax = torch.max(predictions, dim=1)

        # remove backgrounds
        confidscore = confidscore[catargmax!=59]
        proposals = proposals[catargmax!=59]
        catargmax = catargmax[catargmax!=59]

        # change proposals from X,Y,W,H to X,Y,X2,Y2
        proposals[:,2] =  proposals[:,2] + proposals[:,0]
        proposals[:,3] = proposals[:,3] + proposals[:,1]

        for cat in range(59):
            # filter by category and sort by confidence score
            cat_confidence = confidscore[catargmax==cat]
            order = torch.argsort(cat_confidence, descending=True)
            cat_proposals = proposals[catargmax==cat][order]
            cat_confidence[order]

            # remove unneeded bb with NMS
            proposal_inds  = nms(cat_proposals,cat_confidence,iou_threshold=0.5)
            cat_proposals  = cat_proposals[proposal_inds]
            cat_confidence = cat_confidence[proposal_inds]
            
            if len(cat_confidence)>0:

                # check which bb are true detections
                iou_threshold = 0.5
                true = true_bb[true_cat==cat+1]
                if len(true)==0:
                    cat_true = torch.tensor([0]*len(cat_proposals))
                else:
                    iou_bb = torchvision.ops.box_iou(cat_proposals, true_bb[true_cat==cat+1])
                    cat_true = iou_bb.max(1)[0]>iou_threshold
                # 
                cat_scores[cat] = torch.cat([cat_scores[cat],cat_confidence])
                cat_trues[cat] = torch.cat([cat_trues[cat],cat_true])

    calc_AP(cat_scores, cat_trues)


if __name__ == '__main__':

    modelpath = os.path.join("models","restfull_netowrk.pdf")
    architecture = Architecture.from_string("resnet18")
    
    model = architecture.create_network()
    model.load_state_dict(torch.load(modelpath))
    model.to(device)

    with torch.no_grad():
        evaluate(model)




