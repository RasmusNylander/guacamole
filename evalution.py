import tqdm
from dataloader import Patches
from torch.utils.data import DataLoader


def evaluate(model):

    for img_num, (proposals, image, true_bb, true_cat) in enumerate(tqdm(train_loader, leave=False, unit="batches", position=1)):
    
        batch_size = 64
        patch_set = Patches(image, proposals)
        patch_loader = DataLoader(Patches, batch_size=batch_size, shuffle=True, num_workers=1)
    
        predictions = []
        for patch in patch_loader:
            predictions.append(model(patch))

        category, confidence = extract_category(predictions)

        

if __name__ == '__main__':

    evaluate(model)




