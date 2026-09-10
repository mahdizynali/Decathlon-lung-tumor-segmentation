import pickle
import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
from models.lit_segmentation_model import LitLungTumorSegModel
from models.segnet import SuperSegNet
from datasets.lung_tumor_dataset import get_dataset
from collections import Counter




BATCH_SIZE = 32
# num_workers = os.cpu_count()
num_workers = 8
TOP_K_CHECKPOINTS = 5
EARLY_STOPPING_PATIENCE = 10
NUM_EPOCHS = 40
PROJECT_NAME = "maze"
PREPROCESSED_INPUT_DIR = "final"
NUM_CLASSES = 2
WARM_START = False
LEARNING_RATE = 0.001
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_THRESHOLD = 1e-4
WEIGHTS_FILE = PREPROCESSED_INPUT_DIR + '/class_balancing_weights.pkl'


def save_class_balancing_weights(neg_weight, pos_weight):
    with open(WEIGHTS_FILE, 'wb') as f:
        pickle.dump({'neg_weight': neg_weight, 'pos_weight': pos_weight}, f)


def load_class_balancing_weights():
    if os.path.exists(WEIGHTS_FILE):
        with open(WEIGHTS_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['neg_weight'], data['pos_weight']
    return None, None

# def class_balancing_weights(training_data):
#     positive_count = 0
#     negative_count = 0

#     for _, mask in tqdm(training_data):
#         if mask.sum() > 0:
#             positive_count += 1
#         else:
#             negative_count += 1

#     total = positive_count + negative_count
#     positive_weight = negative_count / total
#     negative_weight = positive_count / total

#     return negative_weight, positive_weight


def class_balancing_weights(training_data):
    img_class_count = Counter([1 if mask.sum() > 0 else 0 for data, mask in tqdm(training_data)])
    pos_weight = img_class_count[0] / sum(img_class_count.values())
    neg_weight = img_class_count[1] / sum(img_class_count.values())
    return neg_weight, pos_weight


def get_weighted_random_sampler(training_data, neg_weight, pos_weight):
    weighted_list = [pos_weight if mask.sum() > 0 else neg_weight for (_, mask) in training_data]
    return torch.utils.data.sampler.WeightedRandomSampler(weighted_list, len(weighted_list))


def train_model():
    pl.seed_everything(43)


    training_data = get_dataset(PREPROCESSED_INPUT_DIR, data_type='train')
    validation_data = get_dataset(PREPROCESSED_INPUT_DIR, data_type='val')
    neg_weight, pos_weight = load_class_balancing_weights()
    if neg_weight is None or pos_weight is None:
        neg_weight, pos_weight = class_balancing_weights(training_data)
        save_class_balancing_weights(neg_weight, pos_weight)

    sampler = get_weighted_random_sampler(training_data, neg_weight, pos_weight)

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, num_workers=num_workers, sampler=sampler)
    validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, num_workers=num_workers, shuffle=False)
    # exit(0)
    

    with open(PREPROCESSED_INPUT_DIR + "/train.pkl", 'wb') as f:
        pickle.dump(train_dataloader, f)

    with open(PREPROCESSED_INPUT_DIR + "/val.pkl", 'wb') as f:
        pickle.dump(validation_dataloader, f)

    print("Data loaders saved successfully.")


    # with open(PREPROCESSED_INPUT_DIR + "/train.pkl", 'rb') as f:
    #     train_dataloader = pickle.load(f)

    # with open(PREPROCESSED_INPUT_DIR + "/val.pkl", 'rb') as f:
    #     validation_dataloader = pickle.load(f)

    # print("Data loaders loaded successfully.")


    # print(f"There are {len(training_data)} train images and {len(validation_data)} val images")

    segnet = SuperSegNet()
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.Tensor([neg_weight, pos_weight]))
    model = LitLungTumorSegModel(segnet, loss_fn, NUM_CLASSES, LEARNING_RATE, LR_SCHEDULER_PATIENCE, LR_SCHEDULER_THRESHOLD)

    learning_rate_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor='Val IOU', save_top_k=TOP_K_CHECKPOINTS, mode='max')
    early_stopping_callback = EarlyStopping(monitor='Val Loss', min_delta=1e-5, patience=EARLY_STOPPING_PATIENCE)


    gpus = 1 if torch.cuda.is_available() else 0
    tb_logger = TensorBoardLogger("tb_logs", name="lung_tumor_segmentation")


    trainer = pl.Trainer(gpus=gpus, log_every_n_steps=1, logger = tb_logger,
                         callbacks=[checkpoint_callback, early_stopping_callback,
                                    learning_rate_callback], max_epochs=NUM_EPOCHS)

    print("\nTraning is being start ...\n")
    trainer.fit(model, train_dataloader, validation_dataloader)


def main():

    train_model()


if __name__ == '__main__':
    main()