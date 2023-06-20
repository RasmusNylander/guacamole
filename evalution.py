import tqdm
import torch
from dataloader import Patches, DatasetType, ProposalsEval
from torch.utils.data import DataLoader
import os
from networks import Architecture
from assert_gpu import assert_gpu
device = assert_gpu()


def evaluate(model):

    proposal_set = ProposalsEval(root_dir= "data_wastedetection",ds_type = DatasetType.train)
    img_loader = DataLoader(proposal_set, batch_size=1, shuffle=False, num_workers=1)

    #for img_num, (proposals, image, true_bb, true_cat) in enumerate(tqdm(img_loader, leave=False, unit="batches", position=1)):
    for (proposals, image, true_bb, true_cat) in img_loader:
    
        batch_size = 64
        patch_set = Patches(image, proposals)
        patch_loader = DataLoader(patch_set, batch_size=batch_size, shuffle=False, num_workers=1)
    
        predictions = []
        for patch in patch_loader:

            input_tensor = torch.rand(1000, 4)

            scores = torch.rand(1000, 60)
            #predictions.append(model(patch))
        predictions = torch.tensor(predictions)

        #category, confidence = extract_category(predictions)

        # remove backgrounds
        category = category[category!=60]
        #category = confidence[category~=60]

        for cat in torch.unique(category):
            print(cat)



if __name__ == '__main__':

    """
    modelpath = os.path.join("models","restfull_netowrk.pdf")
    architecture = Architecture.from_string("resnet18")
    
    model = architecture.create_network()
    model.load_state_dict(torch.load(modelpath))
	model.to(device)
    """

    model = None
    evaluate(model)




