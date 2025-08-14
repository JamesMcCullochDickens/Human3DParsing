import torch
from torch import nn
import sys
sys.path.append('/home/zh/human/Human3DSeg')
from Evaluation.pc_seg_eval import pc_seg_eval
import numpy as np

# 1. Create a dummy model
class DummyModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, x):
        # Handle different input types
        if isinstance(x, tuple) and len(x) > 0:
            x = x[0]  # Extract from tuple
            
        # Handle list input
        if isinstance(x, list) and len(x) > 0:
            x = x[0]  # Extract from list
        
        # Match dimensions
        batch_size, num_points = x.shape[:2]
        return torch.rand((batch_size, num_points, self.num_classes), device=x.device)

# 2. Create dummy data with correct dimensions
batch_size = 2
num_points = 1000
num_classes = 3

# Create point cloud data
points = torch.rand((batch_size, num_points, 3))  # [batch_size, num_points, features]
labels = torch.randint(0, num_classes, (batch_size, num_points))  # [batch_size, num_points]

# Fix the device mismatch issue
def custom_preprocess(inp, device_id):
    """Custom preprocessing that properly handles device movement"""
    if isinstance(inp, tuple):
        # Handle tuple input
        return custom_preprocess(inp[0], device_id)
    elif isinstance(inp, list):
        if len(inp) == 1:
            if isinstance(inp[0], list):
                # Handle nested list
                return [tensor.to(device_id) for tensor in inp[0]]
            else:
                # Handle single tensor in list
                return inp[0].to(device_id)
    elif isinstance(inp, torch.Tensor):
        return inp.to(device_id)
    
    # If we can't handle it, pass it through
    return inp

# Create a dataset that properly matches the expected format
class CustomDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # Return in a format that works with the data[:-1], data[-1] expected structure
        return [[points], labels]

# Create dataloader with proper format
dummy_dl = torch.utils.data.DataLoader(
    CustomDataset(),
    batch_size=1,
    collate_fn=lambda batch: batch[0]  # Don't add extra batch dimension
)

# 3. Setup the evaluation dictionary
test_dict = {
    "model": DummyModel(num_classes),
    "cm_num_to_str": {0: "background", 1: "person", 2: "chair"},
    "preprocess_func": custom_preprocess  # Use our custom preprocessor
}

# Print diagnostic information
print("Input format checking:")
for data in dummy_dl:
    print(f"Data type: {type(data)}")
    print(f"Data length: {len(data)}")
    
    # Verify we have the expected structure: data[:-1] and data[-1]
    inp = data[:-1]
    gt = data[-1]
    
    print(f"Input type: {type(inp)}")
    print(f"Input shape or structure: {type(inp[0]) if len(inp) > 0 else 'empty'}")
    
    # Inspect more deeply
    if isinstance(inp, tuple) and len(inp) > 0:
        if isinstance(inp[0], list):
            print(f"Input[0] contents: {[tensor.shape for tensor in inp[0]]}")
        elif isinstance(inp[0], torch.Tensor):
            print(f"Input[0] shape: {inp[0].shape}")
    
    print(f"GT shape: {gt.shape}")
    print(f"GT device: {gt.device}")

# 4. Run evaluation
if __name__ == "__main__":
    try:
        result = pc_seg_eval(test_dict, dummy_dl)
        print(f"Mean IoU: {result}%")
    except Exception as e:
        import traceback
        print(f"\nError: {str(e)}")
        traceback.print_exc()