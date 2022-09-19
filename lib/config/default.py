from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------
# MODEL
# -----------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.DEVICE = 'cuda'
_C.MODEL.DEVICE_IDS = [0]
_C.MODEL.SEED = 1
_C.MODEL.PRETRAINED = False
_C.MODEL.PRETRAIN_PATH = ''
_C.MODEL.DROPOUT = 0.5
_C.MODEL.NORM = 'none'
_C.MODEL.INIT = 'norm'
_C.MODEL.INIT_GAIN = 0.02
_C.MODEL.CONSIST = CN()
_C.MODEL.CONSIST.G = 'resnet_9blocks'
_C.MODEL.CONSIST.D = 3


# -----------------------------------------------------------
# LOSS
# -----------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.NAME = ['gan_loss', 'L1Loss']
_C.LOSS.LAMBDA_A = 10
_C.LOSS.LAMBDA_B = 10
_C.LOSS.LAMBDA_IDENTITY = 0.5

# -----------------------------------------------------------
# INPUT
# -----------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.TYPE = 'image'
_C.INPUT.CHANNEL = 3
_C.INPUT.SIZE = [224, 224]
_C.INPUT.GRAY_MEAN = [0.5]
_C.INPUT.GRAY_STD = [0.5]
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]
_C.INPUT.MODALITY = 'RGB'
_C.INPUT.DIRECTION = True

# -----------------------------------------------------------
# PROCESS
# -----------------------------------------------------------
_C.PROCESS = CN()
_C.PROCESS.RESIZE = True
_C.PROCESS.CROP = True
_C.PROCESS.CROP_SIZE = [256, 256]
_C.PROCESS.FLIP = True
_C.PROCESS.FLIP_P = 0.5
_C.PROCESS.TOTENSOR = True

# -----------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------
_C.OUTPUT = CN()
_C.OUTPUT.TYPE = 'image'
_C.OUTPUT.CHANNEL = 3
_C.OUTPUT.SIZE = [224, 224]

# -----------------------------------------------------------
# DATASET
# -----------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.NAME = ''
_C.DATASET.TYPE = ''
_C.DATASET.ROOT_DIR = ''
_C.DATASET.MAX_SIZE = 'inf'

# -----------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.BATCH_SIZE = 128

# -----------------------------------------------------------
# SOLVER
# -----------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"
_C.SOLVER.LR_SCHEDULER = 'linear'
_C.SOLVER.BASE_LR = 0.0002
_C.SOLVER.MAX_EPOCH = 400
# the epoch that lr decay to 0
_C.SOLVER.LR_DECAY_EPOCH = 200
_C.SOLVER.START_EPOCH = 0
_C.SOLVER.LR_INIT_EPOCH = 200
_C.SOLVER.LR_DECAY_ITERS =  50

# -----------------------------------------------------------
# TRAIN
# -----------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.IS_TRAIN = True

# -----------------------------------------------------------
# TEST
# -----------------------------------------------------------
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 128
_C.TEST.WEIGHT = ""

# -----------------------------------------------------------
# CHECKPOINT
# -----------------------------------------------------------
_C.CHECKPOINT = CN()
_C.CHECKPOINT.RESUME = 'none'
_C.CHECKPOINT.CHECKNAME = ''
