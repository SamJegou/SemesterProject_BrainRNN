import matplotlib.pyplot as plt
import numpy as np

filename_suffixe = ''

rewards = np.load('train/rewards'+filename_suffixe+'.npy')
steps = np.load('train/steps'+filename_suffixe+'.npy')

t = range(len(rewards))

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Reward', color=color)
ax1.plot(t, rewards, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Step number', color=color)  # we already handled the x-label with ax1
ax2.plot(t, steps, color=color)
ax2.tick_params(axis='y', labelcolor=color)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Training'+filename_suffixe)
plt.grid()
plt.show()

