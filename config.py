# dir with ESC50 data
esc50_path = 'data/esc50'

runs_path = 'results'
# sub-epoch (batch-level) progress bar display
disable_bat_pbar = False
#disable_bat_pbar = True

# do not change this block
n_classes = 50
folds = 5
test_folds = [1, 2, 3, 4, 5]
# use only first fold for internal testing
#test_folds = [1]

# sampling rate for waves
sr = 44100 # default wir können die Hälfte der Schritte anzeigen
#sr = 22050
n_fft = 1024 #default
#n_fft = 2048
n_mels = 128
hop_length = 512
#hop_length = 2048
#n_mfcc = 42
normalization = None
# normalization = 'constant'

model_constructor = "AudioMLP(n_steps=431,\
n_mels=config.n_mels,\
hidden1_size=512,\
hidden2_size=128,\
output_size=config.n_classes,\
time_reduce=1)"


# model_constructor = MFCC_10CNN(in_channels=3*config.n_mfcc, num_classes=config.n_classes),

# ###TRAINING
# ratio to split off from training data
val_size = .2  # could be changed
device_id = 0
batch_size = 32
# in Colab to avoid Warning
num_workers = 2
num_workers = 0
# for local Windows or Linux machine
# num_workers = 6#16
persistent_workers = True
persistent_workers = False
epochs = 200
#epochs = 1
# early stopping after epochs with no improvement
patience = 20
lr = 1e-3
weight_decay = 1e-3
warm_epochs = 10
gamma = 0.8
step_size = 5

# ### TESTING
# model checkpoints loaded for testing
test_checkpoints = ['terminal.pt']  # ['terminal.pt', 'best_val_loss.pt']
# experiment folder used for testing (result from cross validation training)
#test_experiment = 'results/2025-04-07-00-00'
test_experiment = 'results/sample-run'


# Also konzept wie folgt:
# 1. model trainieren
# 2. results ordner schlau bennen
# python .\test_crossval.py .\results/2025-04-07-00-00



# Die frage ist nun wie macht man das ordentlich?
# Endpunkt bei uns ist crossvalidaiton accuracy
# permutationen an parametern / einstellungen / hyper parametern müssen getestet werden

# problem ist dass das config file nicht pro result folder gespeichert wird
# Mögliche Planung: Excel Tabelle mit allen parametern und den Ergebnissen anlegen
# Hydra von Facebook
# https://hydra.cc/docs/intro/