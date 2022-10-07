import os
import torch
import datetime
import torch.nn as nn
import pandas as pd

from tqdm import tqdm
from os.path import join
from dataset import resize_input, train_val_split, LifeSpanDataset
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from dp_model.model_files.sfcn_regression import SFCNRegression
from config import *
from sklearn.model_selection import StratifiedKFold

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


def train_one_epoch(model, train_loader_tqdm, optimizer, scheduler, criterion, device):
	train_loss = []
	for x, _, real_y, _ in train_loader_tqdm:
		x, real_y = x.type(torch.FloatTensor).to(device), real_y.type(torch.FloatTensor).to(device)
		model.train()
		pred = model(x)
		train_batch_loss = criterion(pred, real_y)
		train_batch_loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		scheduler.step()

		train_loss.append(train_batch_loss.detach().clone().cpu())

		# update bar
		train_loader_tqdm.set_postfix(loss=torch.stack(train_loss).mean().item())

	return train_loss

def validation(model, val_loader, device):
	val_loss = []
	with torch.no_grad():
		model.eval()
		for x, _, real_y, _ in val_loader:
			x, real_y = x.to(device), real_y.to(device)
			pred = model(x)
			loss = F.l1_loss(pred, real_y)
			val_loss.append(loss.detach().clone().cpu())
	return val_loss


def training_loops(fold, model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, save_path):
	best_loss = 999
	for epoch in range(epochs):
		train_loader_tqdm = tqdm(train_loader, leave=False, ncols=80)
		train_loader_tqdm.set_description(f"Epoch [{epoch + 1}/{epochs}]")
		train_loss = train_one_epoch(model, train_loader_tqdm, optimizer, scheduler, criterion, device)
		val_loss = validation(model, val_loader, device)

		print('{} Epoch {}, Training loss {:.4f}, Validation loss {:.4f}'.format(
			datetime.datetime.now(), epoch,
			torch.stack(train_loss).mean().item(),
			torch.stack(val_loss).mean().item()
			)
		)

		cur_loss = torch.stack(val_loss).mean().item()
		if cur_loss < best_loss:
			torch.save(model.state_dict(), join(save_path, f'age_prediction_model_fold{fold}.pt'))
			print(f"Loss descreses on validation set\nModel saved to {join(save_path, f'age_prediction_model_fold{fold}.pt')}")
			best_loss = cur_loss


def train():
	device = torch.device('cuda') 
	df = pd.read_csv(METADATA_CSV).reset_index(drop=True)
	kf = StratifiedKFold(10, shuffle=True, random_state=42)
	# resize_input(df, TRAIN_DIR)
	for i, (train_index, val_index) in enumerate(kf.split(df.Age, df.Age.apply(lambda x: int(x / 5)))):
		df_train, df_val = df.iloc[train_index].reset_index(drop=True), df.iloc[val_index].reset_index(drop=True)
		
		train_loader = DataLoader(LifeSpanDataset(TRAIN_DIR, df_train, random_shift=True, flip=True), batch_size=11, shuffle=True)
		val_loader = DataLoader(LifeSpanDataset(TRAIN_DIR, df_val), batch_size=11)
		
		model = DataParallel(SFCNRegression()).to(device)
		criterion = nn.MSELoss()
		optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
		scheduler = StepLR(optimizer, step_size=20*len(train_loader), gamma=1/3)

		os.makedirs(OUTPUT_NETWORK, exist_ok=True)
		training_loops(i, model, criterion, optimizer, scheduler, EPOCHS, train_loader, val_loader, device, OUTPUT_NETWORK)
	input('enter any key to stop')

if __name__ == '__main__':
	train()