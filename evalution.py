import tqdm
import torch
from dataloader import Patches, DatasetType, ProposalsEval
from torch.utils.data import DataLoader


def evaluate(model):

    Proposals = ProposalsEval(ds_type = DatasetType.train)
    img_loader = DataLoader(ProposalsEval, batch_size=1, shuffle=False, num_workers=1)

    for img_num, (proposals, image, true_bb, true_cat) in enumerate(tqdm(img_loader, leave=False, unit="batches", position=1)):
    
        batch_size = 64
        patch_set = Patches(image, proposals)
        patch_loader = DataLoader(Patches, batch_size=batch_size, shuffle=False, num_workers=1)
    
        predictions = []
        for patch in patch_loader:
            predictions.append(model(patch))
        predictions = torch.tensor(predictions)

        #category, confidence = extract_category(predictions)

        # remove backgrounds
        category = category[category~=60]
        #category = confidence[category~=60]

        for cat in torch.unique(category):
            print(cat)



if __name__ == '__main__':

    model = None
    evaluate(model)




