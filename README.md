# Automating Stepsizes for Decentralized Optimization and Learning with Geometric Convergence

Each of the three experiments contain a `trainer.sh` script that can be used to train all runs for the given experiment, which will store all data in corresponding results folders with `.csv` files.

To run an experiment, use command:
```
./trainer.sh
```

### Dependencies

```
torch==2.0.0+cu117
torchaudio==2.0.0+cu117
torchvision==0.15.0+cu117
python==3.11.3
numpy==1.26.4
scipy==1.12.0
pandas==2.2.1
sklearn==1.0.2
matplotlib==3.8.3
```
