Training code for the article Accurate brain age prediction with lightweight deep neural networks https://doi.org/10.1101/2019.12.17.879346

## Examples
Checkout the file [**examples.ipynb**](https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain/blob/master/examples.ipynb)
```python
model = SFCN()
model = torch.nn.DataParallel(model)
# This is to be modified with the path of saved weights
p_ = './run_20190719_00_epoch_best_mae.p'
model.load_state_dict(torch.load(p_))
```


