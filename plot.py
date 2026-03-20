import matplotlib.pyplot as plt

def plot(loss_history):
    epochs = range(1, len(loss_history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_history, marker='o', linestyle='-', color='b', label='Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average Loss vs. Epoch')
    plt.grid(True)
    plt.legend()