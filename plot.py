import pandas as pd
import itertools
import matplotlib.pyplot as plt



def plot_normal_vs_double(normal):
	plt.clf()
	double = pd.read_csv(f'data/32_1_shallow_double.csv')

	plt.plot(normal.iloc[:, 2], label='DQN', color='blue', linewidth=2)
	plt.plot(normal.iloc[:, 1], color='blue', alpha=0.3, linewidth=0.5)
	plt.plot(double.iloc[:, 2], label='DDQN', color='red', linewidth=2)
	plt.plot(double.iloc[:, 1], color='red', alpha=0.3, linewidth=0.5)
	plt.hlines(200,0,1000, label='solved', color='black',  linestyles='dashed')
	plt.title('DQN vs. DDQN')
	plt.ylabel('Reward')
	plt.xlabel('Episode')
	plt.grid()

	plt.legend(loc='lower right')
	plt.savefig('plots/DQNvsDDWN')
	plt.show()



def plot_normal():
	batch_size = [32, 64, 128]
	target_update = [1, 10, 20]
	network = ['shallow', 'deep']
	averages = []
	names = []



	for experiment in itertools.product(batch_size, target_update, network):
		name = str((experiment[0])) + '_' + str((experiment[1])) + '_' + str((experiment[2]))
		df = pd.read_csv(f'data/{name}.csv')

		# get ast index of average reward column
		averages.append(df.iloc[:, 2].iloc[-1])
		names.append(name)
		plt.plot(df.iloc[:, 2], label=name, linewidth=0.8)
		cnt = 1
		for i in df.iloc[:, 2]:
			if i >=200:
				solved = cnt
				break
			cnt+=1
		if name == '32_1_shallow':
			save = df

		print(f'{name}       {round(df.iloc[:, 2].iloc[-1], 1)}       {solved}')

	plt.hlines(200,0,1000, label='solved', color='black',  linestyles='dashed')
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

	plt.plot(df_lowest.iloc[:, 2], label=names[averages.index(lowest_reward)])
	plt.plot(df_highest.iloc[:, 2], label=names[averages.index(highest_reward)])
	plt.legend()
	# plt.show()

	plot_normal_vs_double(save)


def main():
	plot_normal()



if __name__ == '__main__':
	main()