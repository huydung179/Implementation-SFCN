import os
import torch
import datetime

from tqdm import tqdm
from os.path import join
from dataset import resize_input, train_val_split, LifeSpanDataset
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dp_model.dp_loss import my_KLDivLoss
from torch.optim.lr_scheduler import StepLR
from dp_model.model_files.sfcn import SFCN
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES


def train_one_epoch(model, train_loader_tqdm, optimizer, scheduler, criterion, device):
	train_loss = []
	for x, soft_y, real_y, _ in train_loader_tqdm:
		x, soft_y = x.to(device), soft_y.to(device)
		model.train()
		pred = model(x)
		pred = pred.reshape(pred.size(0), pred.size(1))
		train_batch_loss = criterion(pred, soft_y)
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
		for x, soft_y, real_y, bc in val_loader:
			x, real_y, bc = x.to(device), real_y.to(device), bc.to(device)
			soft_y = soft_y.to(device)
			pred = model(x)
			pred = torch.exp(pred).reshape(pred.size(0), pred.size(1))
			pred = pred * bc
			pred = pred.sum(dim=-1, keepdims=True)
			loss = F.l1_loss(pred, real_y)
			val_loss.append(loss.detach().clone().cpu())
	return val_loss


def training_loops(model, criterion, optimizer, scheduler, epochs, train_loader, val_loader, device, save_path):
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
			torch.save(model.state_dict(), join(save_path, 'age_prediction_model.pt'))
			print(f"Loss descreses on validation set\nModel saved to {save_path}")
			best_loss = cur_loss


def train():
	device = torch.device('cuda')
	resize_input(T1_DIR, TRAIN_DIR)
	df_train, df_val = train_val_split(METADATA_CSV)

	train_set = LifeSpanDataset(TRAIN_DIR, df_train, random_shift=True, flip=True)
	train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
	val_loader = DataLoader(LifeSpanDataset(TRAIN_DIR, df_val), batch_size=8)
	model = DataParallel(SFCN()).to(device)
	criterion = my_KLDivLoss
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
	scheduler = StepLR(optimizer, step_size=30*len(train_loader), gamma=0.3)

	os.makedirs(OUTPUT_NETWORK, exist_ok=True)
	training_loops(model, criterion, optimizer, scheduler, EPOCHS, train_loader, val_loader, device, OUTPUT_NETWORK)

if __name__ == '__main__':
	train()