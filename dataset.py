import os
import glob
import torch
import numpy as np
import pandas as pd
import nibabel as nib

from tqdm import tqdm
from os.path import join, basename, exists
from torch.nn import functional as F
from torch.utils.data import Dataset
from dp_model.dp_utils import num2vect


def resize_input(metadata, out_dir, target_shape=[181, 217, 181], device=None, s=2):
	os.makedirs(out_dir, exist_ok=True)
	for i in tqdm(range(len(metadata)), leave=False, ncols=80):
		input_tensor = torch.from_numpy(np.array(nib.load(metadata.iloc[i].T1_path).get_fdata()).astype('float32')).to(device)
		input_tensor = F.interpolate(input_tensor.unsqueeze(0).unsqueeze(0), size=target_shape).squeeze()
		torch.save(input_tensor.cpu(), join(out_dir, metadata.iloc[i].Subject + '.pt'))


def train_val_split(metadata_csv, train_ratio=0.8):
	df_total = pd.read_csv(metadata_csv)
	df_train = pd.DataFrame(columns=df_total.columns)
	for age_group in df_total.age_group.unique():
		df_train = pd.concat([
			df_train, df_total[df_total.age_group == age_group
		].sample(frac=train_ratio, replace=False)], ignore_index=True).reset_index(drop=True)

	df_val = df_total[~df_total.filename.isin(df_train.filename)].reset_index(drop=True)
	return df_train, df_val


class LifeSpanDataset(Dataset):
	def __init__(self, resized_img_dir, metadata, input_shape=[167, 212, 160], random_shift=False, s=2, flip=False):
		super(LifeSpanDataset, self).__init__()
		self.metadata = metadata
		self.resized_img_dir = resized_img_dir
		self.input_shape = input_shape
		self.random_shift = random_shift
		self.s = s
		self.flip = flip
		self.border = [
			(181 - input_shape[0]) // 2,
			(217 - input_shape[1]) // 2,
			(181 - input_shape[2]) // 2]
	
	def __len__(self):
		return len(self.metadata)

	def __getitem__(self, i):
		
		if self.random_shift:
			start = torch.randint(low=-self.s, high=self.s + 1, size=(3,))
			start = [e.item() + b for e, b in zip(start, self.border)]
		else:
			start = self.border
		input_img = torch.load(join(self.resized_img_dir, self.metadata.Subject.iloc[i]) + '.pt').unsqueeze(0)
		input_img = input_img / input_img.mean()
		input_img = input_img[
			...,
			start[0]:start[0] + self.input_shape[0],
			start[1]:start[1] + self.input_shape[1],
			start[2]:start[2] + self.input_shape[2]
		]

		
		if self.flip:
			flip = torch.rand(1) > 0.5
			if flip:
				input_img = torch.flip(input_img, dims=[-3])
				# T1_img = nib.load(join('/opt/deep/data/datasets/lifespan1_CN/CN/t1', 'n_mmni_f' + self.metadata.filename.iloc[i]))
				# img = nib.Nifti1Image(input_img.squeeze().cpu().numpy().astype("float32"), T1_img.affine)
				# img.to_filename(join('/opt/deep/data', f"{self.metadata.filename.iloc[i]}_cp.nii.gz"))
				# raise NotImplementedError
		real_y = torch.tensor([self.metadata.Age.iloc[i]])
		age_soft, bc = num2vect(self.metadata.Age.iloc[i])
		age_soft = torch.tensor(age_soft).type(torch.FloatTensor)
		bc = torch.from_numpy(bc)

		return input_img, age_soft, real_y, bc