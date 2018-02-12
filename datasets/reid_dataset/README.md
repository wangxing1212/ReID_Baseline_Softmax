## Person Re-Identification Datasets Downloader
Currently Support Datasets: <br />
VIPeR <br />
CUHK01 <br />
CUHK03 <br />
Market1501 <br />
Market1501Attribute <br />
DukeMTMC <br />
DukeMTMCAttribute

More datasets will be added in the future.
### Require Packages
```bash
sudo pip install -r requirements.txt
sudo pip3 install -r requirements.txt
```

### How to Use
Import function:
```python
from reiddataset import download_dataset
```
Simple Example:
```python
file_id = "CUHK01" #VIPeR, CUHK02, CUHK03
destination = "./dataset" #Dataset Directory
download_dataset(file_id, destination)
```
or
```bash
python download_dataset.py Market1501 ~/Datasets
```

Import function:
```python
from reiddataset import import_VIPeR
from reiddataset import import_CUHK01
from reiddataset import import_CUHK03
```
Simple Example:
for import_CUHK03(dataset_dir, detected = False) change detected to True if used detected images
```python
dataset_dir = '/home/linshan/dataset/'
cuhk03 = import_CUHK03(dataset_dir)
```
