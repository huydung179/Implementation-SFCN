import os
import torch
import pickle
import pandas as pd

from tqdm import tqdm
from os.path import join
from dataset import resize_input, LifeSpanDataset
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dp_model.model_files.sfcn_regression import SFCNRegression
from sklearn.metrics import r2_score
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

def validation(model, dataloader, device):
	preds, gt = [], []
	with torch.no_grad():
		model.eval()
		for i, (x, soft_y, real_y, bc) in enumerate(tqdm(dataloader, leave=False, ncols=80)):
			x, real_y = x.to(device), real_y.to(device)
			pred = model(x)
			preds.append(pred.cpu())
			gt.append(real_y.cpu())
	preds = torch.cat(preds, dim=0)
	gt = torch.cat(gt, dim=0)
	return preds, gt


def evaluation():
	device = torch.device('cuda')
	df = pd.read_csv(METADATA_CSV_TEST)
	df = df[df.Dataset.isin(['UKBB'])].reset_index(drop=True)
	# df = df[df.Subject.str[:5] == "ABIDE"].reset_index(drop=True)
	# resize_input(df, TEST_DIR)
	dataset = LifeSpanDataset(TEST_DIR, df)
	dataloader = DataLoader(dataset, batch_size=4)

	model = DataParallel(SFCNRegression()).to(device)
	model.load_state_dict(torch.load(join(OUTPUT_NETWORK, "age_prediction_model_regression.pt")))
	os.makedirs(OUTPUT_EVAL, exist_ok=True)
	preds, gt = validation(model, dataloader, device)
	print(F.l1_loss(preds, gt))
	print(r2_score(gt.cpu().numpy(), preds.cpu().numpy()))
	with open(join(OUTPUT_EVAL, "out_regression.pkl"), "wb") as f:
		pickle.dump([preds, gt, F.l1_loss(preds, gt)], f)

if __name__ == '__main__':
	evaluation()