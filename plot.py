import pandas as pd
import itertools
import matplotlib.pyplot as plt



def plot_normal():
	batch_size = [32, 64]
	target_update = [1, 10]
	network = ['shallow']
	averages = []
	names = []



	for experiment in itertools.product(batch_size, target_update, network):
		name = str((experiment[0])) + '_' + str((experiment[1])) + '_' + str((experiment[2]))
		df = pd.read_csv(f'data/{name}.csv')

		# get ast index of average reward column
		averages.append(df.iloc[:, 2].iloc[-1])
		names.append(name)
		plt.plot(df.iloc[:, 2], label=name, linewidth=0.8)
		print(f'{name}       {round(df.iloc[:, 2].iloc[-1], 1)}')

	plt.title('DQN Hyperparameter experiments')
	plt.ylabel('Reward')
	plt.xlabel('Episode')
	plt.grid()

	plt.legend(ncol=2)
	plt.savefig('plots/DQN_hyperparams')
	plt.show()



	# get highest and lowest reward
	lowest_reward = min(averages)
	highest_reward = max(averages)

	# plot highest vs lowest
	df_lowest = pd.read_csv(f'data/{names[averages.index(lowest_reward)]}.csv')
	df_highest = pd.read_csv(f'data/{names[averages.index(highest_reward)]}.csv')

	# plt.plot(df_lowest.iloc[:, 2], label=names[averages.index(lowest_reward)])
	# plt.plot(df_highest.iloc[:, 2], label=names[averages.index(highest_reward)])
	# plt.legend()
	# plt.show()


def main():
	plot_normal()



if __name__ == '__main__':
	main()