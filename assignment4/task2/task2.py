import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    # Compute intersection
    if (prediction_box[0] > gt_box[2] or prediction_box[2] < gt_box[0] or prediction_box[1] > gt_box[3]
            or prediction_box[3] < gt_box[1]):
        return 0
    y_dim_inter = min(prediction_box[3], gt_box[3]) - max(prediction_box[1], gt_box[1])
    x_dim_inter = min(prediction_box[2], gt_box[2]) - max(prediction_box[0], gt_box[0])
    A_inter = y_dim_inter * x_dim_inter

    A_uni = (prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1]) \
            + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) - A_inter

    # Compute union
    iou = A_inter / A_uni
    assert iou >= 0 and iou <= 1
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    if num_tp + num_fp == 0:
        return 1
    return num_tp / (num_tp + num_fp)


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    if num_tp + num_fn == 0:
        return 0
    return num_tp / (num_tp + num_fn)



def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    iou_scores = [[] for i in range(len(prediction_boxes))]
    for i, pred_box in enumerate(prediction_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            iou_scores[i].append(iou * (iou > iou_threshold))

    # Sort all matches on IoU in descending order

    iou_flat = iou_scores.copy() #[item for sublist in iou_scores for item in sublist]
    iou_flat = [item for sublist in iou_flat for item in sublist]
    iou_flat.sort(reverse=True)
    #print("Sorted IOU scores: " + str(iou_flat))

    # Find all matches with the highest IoU threshold
    return_pred_boxes = np.empty((0, 4))
    return_gt_boxes = np.empty((0, 4))

    ar = np.array(iou_scores)
    prediction_boxes = np.array(prediction_boxes)
    gt_boxes = np.array(gt_boxes)

    list_i = []
    list_j = []
    for idx_val, val in enumerate(iou_flat):
        i, j = np.where(ar==val)
        #print("list_i: " + str(list_i))
        #print("list_j: " + str(list_j))
        #print("val: " + str(val))
        if i[0] in list_i or j[0] in list_j or val == 0:
            continue
        list_i.append(i[0])
        list_j.append(j[0])
        return_pred_boxes = np.vstack((return_pred_boxes, prediction_boxes[i[0]]))
        #print("Return_pred_boxes: " + str(return_pred_boxes))
        return_gt_boxes = np.vstack((return_gt_boxes, gt_boxes[j[0]]))
        #print("Return_gt_boxes: " + str(return_gt_boxes))


    return return_pred_boxes, return_gt_boxes


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """
    matched_pred, matched_gt = get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold)
    dict = {}
    dict['true_pos'] = len(matched_pred)
    dict['false_pos'] = prediction_boxes.shape[0] - dict['true_pos']
    dict['false_neg'] = gt_boxes.shape[0] - dict['true_pos']

    return dict


def calculate_precision_recall_all_images(
        all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    prec = 0
    recall = 0
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    n = len(all_prediction_boxes)
    for idx, pred in enumerate(all_prediction_boxes):
        dict = calculate_individual_image_result(all_prediction_boxes[idx], all_gt_boxes[idx], iou_threshold)
        tot_tp += dict['true_pos']
        tot_fp += dict['false_pos']
        tot_fn += dict['false_neg']

    prec = calculate_precision(tot_tp, tot_fp, tot_fn)
    recall = calculate_recall(tot_tp, tot_fp, tot_fn)

    return prec, recall


def get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold
):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.


       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = []
    recalls = []

    for idx, conf in enumerate(confidence_thresholds):
        pred_boxes = all_prediction_boxes.copy()
        for img_idx, img in enumerate(pred_boxes):
            to_delete = []
            #print("="*80)
            for box_idx, box in enumerate(img):
                if confidence_scores[img_idx][box_idx] < conf:
                    to_delete.append(box_idx)
            #print(str(to_delete) + " ----- %f"%conf)

            pred_boxes[img_idx] = np.delete(pred_boxes[img_idx][:], to_delete, axis=0)

        prec, rec = calculate_precision_recall_all_images(pred_boxes, all_gt_boxes, iou_threshold)
        precisions.append(prec)
        print(prec)
        recalls.append(rec)


    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0, 1.0, 11)
    # YOUR CODE HERE

    print("Precision Sum: " + str(sum(precisions)))
    print("Recalls Sum: " + str(sum(recalls)))

    average_precision = 0
    interpolated_precision = [np.amax(np.flipud(precisions)[i:]) for i in range(len(recalls))]
    interpolated_precision = np.flipud(interpolated_precision)

    for rcl_lvl in recall_levels:
        max_prec = 0
        for idx, rcl in enumerate(recalls):
            if rcl >= rcl_lvl and precisions[idx] > max_prec:
                max_prec = precisions[idx]
        average_precision += max_prec/len(recall_levels)


    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)
