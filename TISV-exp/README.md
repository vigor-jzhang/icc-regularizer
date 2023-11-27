## Pipelines for TI-SV experiment

0. Download VoxCeleb 1 and 2 dataset. Then using 'gen_train_test.py' to generate the training and testing dataset indexing file.

1. Change working folder to current folder:

```
cd [path]/[to]/TISV-exp/
```

2. Create conda env with PyThon 3.8:

```
conda create -n icc_tisv python=3.8
conda activate icc_tisv
```

3. Install required packages by using requirements.txt:

```bash
pip install -r requirements.txt
```

4. Modify the config.yaml with correct path:

```yaml
train_dir: 'path/to/vox_train.pickle'
test_dir: 'path/to/vox_test.pickle'
```

And also you can modify other configuration parameters.

5. Run experiment of GE2E loss only:

```
python train_ge2e_only.py
```

6. Run experiment of ICC loss plus GE2E loss:

```
python train_icc_ge2e.py
```