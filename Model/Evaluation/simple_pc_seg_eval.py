'''Introduction'''
# I am going to document the learning path of evaluating point cloud segmentation results here.
# Rather than rewriting the entire script, I will focus on the key parts and concepts that are 
# important for understanding the evaluation process.

# Understand the core principles - Focus on the key concepts:
    # How metrics like IoU are calculated
    # How batch processing works
    # How results are aggregated

# Implement a simplified version - Create a small version focusing just on the essential parts:
# Modify specific components - For deeper understanding, try variations:
    # Add different metrics (precision, recall)
    # Modify how class weights are handled
    # Implement confusion matrix visualization

# Why this approach is better:
    # Evaluation code is fairly standardized - The fundamentals (TP, FP, FN, IoU) are consistent across projects
    # Implementation details may vary - But understanding the core concepts transfers well
    # Active learning is more effective - Making changes to test your understanding beats passive copying

# To internalize it:
    # Use it on toy examples - Create small test cases and verify calculations manually
    # Explain it to someone else - Teaching reinforces understanding
    # Trace through with visualizations - Draw what happens with a small batch of data

"""simplified point cloud segmentation evaluator"""
import torch

def calculate_metrics(pred, gt, num_classes):
    """A simplified point cloud segmentation evaluator"""
    results = {}
    
    # Initialize metrics
    for class_id in range(num_classes):
        results[class_id] = {"tp": 0, "fp": 0, "fn": 0}
    
    # Calculate per-class metrics
    for class_id in range(num_classes):
        # Create binary masks
        pred_mask = (pred == class_id)
        gt_mask = (gt == class_id)
        
        # Calculate TP, FP, FN
        tp = torch.logical_and(pred_mask, gt_mask).sum().item()
        fp = torch.logical_and(pred_mask, ~gt_mask).sum().item()
        fn = torch.logical_and(~pred_mask, gt_mask).sum().item()
        
        results[class_id]["tp"] = tp
        results[class_id]["fp"] = fp
        results[class_id]["fn"] = fn
        
        # Calculate IoU
        if tp + fp + fn > 0:
            results[class_id]["iou"] = tp / (tp + fp + fn)
        else:
            results[class_id]["iou"] = 0.0
    
    # Calculate mIoU
    valid_classes = 0
    iou_sum = 0
    for class_id in range(num_classes):
        if results[class_id]["tp"] + results[class_id]["fn"] > 0:
            valid_classes += 1
            iou_sum += results[class_id]["iou"]
    
    results["miou"] = iou_sum / valid_classes if valid_classes > 0 else 0
    
    return results