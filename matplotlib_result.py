import matplotlib.pyplot as plt
import  matplotlib.ticker
import csv

log = 'set5_Adam_TP5_sharp.csv'
# set5_SGD_TP6_sharp.csv
# set5_Adam_TP5_sharp.csv

with open('past_logs/{}'.format(log), 'r') as csvfile:
    epochs = []
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    plots = csv.reader(csvfile, delimiter=',')
    counter = 0
    for row in plots:
        if counter != 0:
            epochs.append(int(row[0]))
            train_loss.append(float(row[1]))
            train_acc.append(float(row[2]))
            valid_loss.append(float(row[3]))
            valid_acc.append(float(row[4]))

        counter += 1

plt.figure()
plt.grid(b=True, which='major', color='#666666', linestyle='-', alpha=0.2)
plt.xlabel('epoch')
plt.xlim(0, 105)
#
# plt.ylim(0.54, 0.74)
# plt.plot(epochs, train_loss, label='train', linewidth=1)
# plt.plot(epochs, valid_loss, label='validation', linewidth=1)
# plt.ylabel('cross entropy loss')

plt.ylim(0.3, 1.0)
plt.plot(epochs, train_acc, label='train', linewidth=1)
plt.plot(epochs, valid_acc, label='validation', linewidth=1)
plt.ylabel('accuracy')

# plt.title('{}'.format('Sharpen T5 Adam'))
plt.legend()
# type = 'loss'
type = 'accuracy'
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.3f}"))
# plt.show()
plt.savefig('plot/TP3-1_{}.png'.format(type))