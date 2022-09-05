CHECKPOINT_DIR = "models"
device = 'gpu:0'
MVTEC_CATEGORIES = [
    "toothbrush",
    "screw",
    "hazelnut",
    "transistor",
    "tile",
    "pill",
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "leather",
    "metal_nut",
    "wood",
    "zipper",
]

BACKBONE_RESNET18 = "resnet18"

SUPPORTED_BACKBONES = [
    BACKBONE_RESNET18,
]

TRAIN_BATCH_SIZE = 32 #32#60 #70 #32
BATCH_SIZE = TRAIN_BATCH_SIZE
TEST_BATCH_SIZE = 40#60
GRADIENT_SUM = 1#6
NUM_EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 10
EVAL_INTERVAL = 1
CHECKPOINT_INTERVAL = 1


PIXEL_AUROC_RATIO = 0.5
IMAGE_AUROC_RATIO = 0.5
PATIENCE = 30

