import matplotlib.pyplot as plt
PLOT_PATH = "plot.png"
# plot the training loss
import pickle
def load_dict(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
H=load_dict("H")
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(PLOT_PATH)