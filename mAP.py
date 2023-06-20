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


def Precision_recall_curve_plot(precisions,recalls, AP, path, num, f1):
    plt.plot(recalls, precisions, linewidth=4, color="red", zorder=0)
    plt.scatter(recalls[the_maximum_index(f1)], 
                precisions[the_maximum_index(f1)], 
                zorder=1, linewidth=6, label="Max f1-score point")

    plt.xlabel("Recall", fontsize=12, fontweight='bold')
    plt.ylabel("Precision", fontsize=12, fontweight='bold')
    plt.legend(['AP value: ' + "%.4f" % AP, 'Maximum f1-score: ' + str(max(f1))], loc="upper right")
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/test_{num}.png")
    
def calc_AP(group_pred_scores, group_y_true):
    path = './precision-recall_curve'
    APs = []
    
    for i in range(59):
        num = 1 + i  #TODO: Input the number for the precision-recall number if you want to generate the graphs
        
        pred_scores = group_pred_scores[i].tolist() #TODO: The prediction input
        y_true = group_y_true[i].tolist() #TODO: Input the true
        y_true = ["positive" if value == 1 else "negative" for value in y_true]

        
        thresholds = np.arange(start=0.2, stop=1.0, step=0.05)

        precisions, recalls = precision_recall_curve(y_true=y_true, pred_scores=pred_scores,
                                                    thresholds=thresholds)

        f1 = f1_score(precisions, recalls)  #Calclate the buest tradeoff point from the pairs of precisions and recalls.
        
        #One category situation
        AP = AP_cal(precisions, recalls)
        APs.append(AP)  # The list will contain various APs
        
        #Store the image of precision-recall curve
        Precision_recall_curve_plot(precisions,recalls, AP, path, num, f1)
         
    '''
    #IoU related testing
    gt_box = [320, 220, 680, 900] #Pretend the position of ground box will be performed a list
    pred_box = [500, 320, 550, 700]  #Pretend the position of prediction box will be performed a list
    iou, intersect, union = intersection_over_union(gt_box, pred_box)
    print('IoU: {iou} \nIntersect: {intersect} \nUnion: {union}')
    '''
    # mAP: Mean Average Precision (for multiple categories)
    mAP = mAP_cal(APs)
    print('Calculated mAP: ' ,mAP)
    return APs
    
