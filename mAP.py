import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as pc
import os

def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
        
        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def the_maximum_index(f1):
    for i in range(len(f1)):
        if f1[i] == max(f1):
            return i

def f1_score(precisions, recalls):
    f1 = 2*((np.array(precisions)*np.array(recalls)) / (np.array(precisions)+np.array(recalls)))
    return f1

def AP_cal(precisions, recalls):
    AP = np.sum((np.array(recalls[:-1]) - np.array(recalls[1:])) * np.array(precisions[:-1]))
    return AP

def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), 
                          max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), 
                              min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection
    
    iou = intersection / union

    return iou, intersection, union

def mAP_cal(ap_values):
    mAP = sum(ap_values)/len(ap_values)
    return mAP
    


if __name__ == '__main__':
    path = './precision-recall_curve'
    num = 1 #TODO: Input the number for the precision-recall number if you want to generate the graphs
    
    #For testing 
    pred_scores = [0.7, 0.3, 0.5, 0.6, 0.55, 0.9, 0.4, 0.2, 0.4, 0.3]  #TODO: The prediction input
    y_true = ["positive", "negative", "negative", "positive", "positive", 
            "positive", "negative", "positive", "negative", "positive"] #TODO: Input the true 
    
    threshold = 0.5
    y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
    #print(y_pred)

    r = np.flip(sklearn.metrics.confusion_matrix(y_true, y_pred))
    
    
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
    thresholds = np.arange(start=0.2, stop=0.7, step=0.05)

    precisions, recalls = precision_recall_curve(y_true=y_true, pred_scores=pred_scores,
                                                thresholds=thresholds)

    f1 = f1_score(precisions, recalls)  #Calclate the buest tradeoff point from the pairs of precisions and recalls.
    
    #Store the image of precision-recall curve
    plt.plot(recalls, precisions, linewidth=4, color="red", zorder=0)
    plt.scatter(recalls[the_maximum_index(f1)], 
                precisions[the_maximum_index(f1)], 
                zorder=1, linewidth=6, label="Max f1-score point")

    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.legend(loc="upper right")
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    os.makedirs(path, exist_ok=True)
    plt.savfig(f"{path}/test_{num}.png")

    #One category situation
    AP = AP_cal(precisions, recalls)
    print(AP)
    
    #IoU related testing
    gt_box = [320, 220, 680, 900] #Pretend the position of ground box will be performed a list
    pred_box = [500, 320, 550, 700]  #Pretend the position of prediction box will be performed a list
    iou, intersect, union = intersection_over_union(gt_box, pred_box)
    print('IoU: {iou} \nIntersect: {intersect} \nUnion: {union}')
    
    # mAP: Mean Average Precision (for multiple categories)
    l_APs = [] # The list of various APs
    mAP = mAP_cal(l_APs)
    print('Mean Average Precision: ' + str(mAP))
    
    
    
    
    


