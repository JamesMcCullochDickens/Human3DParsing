# data_fps = {"Human_3D_Models": "/home/zh/thuman_dataset/thuman2/THuman2.1_Release/model",
#             "Renders_Outer": "/home/zh/human/Human3DSEG/data/processed/render_im",
#             "M2FP_Parsing_path": "/home/zh/human/m2fp",
#             "Human_Seg_PC_Data": "/home/zh/human/Human3DSeg/Data/PC_Data"}
import os

# Get the base directory (workspace root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to the Human3DSEG directory

data_fps = {
    "Human_3D_Models": os.path.join(BASE_DIR, "data/raw"),
    "Renders_Outer": os.path.join(BASE_DIR, "data/processed/render_im"),
    # "M2FP_Parsing_path": os.path.join(BASE_DIR, "Data_Processing/src/m2fp"),
    "M2FP_Parsing_path": os.environ.get("M2FP_PATH", "/home/zh/human/m2fp"),
    "Human_Seg_PC_Data": os.path.join(BASE_DIR, "data/processed")
}