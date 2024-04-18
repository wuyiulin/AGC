import pandas as pd
import matplotlib.pyplot as plt
import pdb

def vision(csv_path):
    train_log = pd.read_csv(csv_path)

    # get each epoch loss
    loss_list = []
    Loss = train_log['Loss']
    epochs = train_log['Epoch'].iloc[-1]

    for e in range(int(epochs)):
        e_loss = Loss.iloc[e]
        loss_list.append(round(e_loss, 3))

    # Draw zone
    x = range(1, epochs+1)
    y = loss_list
    plt.xlabel('epochs', fontsize="10")
    plt.ylabel('Loss', fontsize="10")
    plt.title('AutoEncoder Loss', fontsize="18")
    plt.plot(x, y, color='red', linestyle="-", linewidth="2", markersize="3", marker=".")
    plt.show()

if __name__ == '__main__':
    csv_path = 'log/AutoEncoder_train_loss.csv'
    vision(csv_path)
