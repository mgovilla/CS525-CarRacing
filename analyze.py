import matplotlib.pyplot as plt

with open('rewards.out', 'r') as f:
    lines = f.readlines()
    total, avgs = 0, []
    for i in range(len(lines)):
        total += float(lines[i])
        if i % 30 == 29:
            avgs.append(total / 30.0)
            total = 0


    plt.plot(avgs)
    plt.xlabel('epoch')
    plt.title('30 episode moving average of rewards')
    plt.ylabel('reward')
    plt.savefig("Overall_Training.png")