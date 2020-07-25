import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

"""
init: k bandits with Q(a) = 0, N(a) = 0
Loop: Select A (explore and exploit)
Run bandit : take action and return a corresponding reward
update N = N+1
Update Q(A) = Q(A)+ ....
"""


class Bandit:
    def __init__(self):
        self.N = 0
        self.Q = np.random.standard_normal()
        self.R = 0

    def play(self):
        self.R = np.random.normal(self.Q, 1)
        self.N += 1
        self.Q = self.Q + (1 / self.N) * (self.R - self.Q)
        return self.R


class K_Bandit:
    def __init__(self, k=10, e=0.1):
        self.bandits = [Bandit() for i in range(k)]
        self.rewards = []
        self.e = e
        self.took_opt = 0

    def get_rewards(self):
        return np.asarray(self.rewards), self.took_opt

    def play_game(self):

        rew = 0
        bandit = self.bandits[0]
        if np.random.binomial(1, self.e):
            bandit = np.random.choice(self.bandits)
            self.rewards.append(bandit.play())
            pass  # take random bandit to play

        else:
            self.took_opt += 1
            for b in self.bandits:
                if b.Q >= rew:
                    rew = b.Q
                    bandit = b
            self.rewards.append(bandit.play())


def run_k_armed_bandit(k=10, e=0.1, runs=1000):
    k_bandit = K_Bandit(k, e)

    for i in range(runs):
        k_bandit.play_game()

    return k_bandit.get_rewards()


def plot_results(results):

    res1 = results["0"]["results"]
    mean1 = mean(res1)
    took1= mean(results["0"]["took_opts"])
    x = [i for i in range(len(res1))]


    plt.plot(x, res1)
    plt.legend([f"y = e=0, mean = {int(mean1)}, exploit_steps={int(took1)}"])
    plt.show()

    res2 = results["0.1"]["results"]
    mean2 = mean(res2)
    took2 = mean(results["0.1"]["took_opts"])

    plt.plot(x, res2)
    plt.legend([f"y = e=0.1, mean = {int(mean2)}, exploit_steps={int(took2)}"])
    plt.show()

    res3 = results["0.01"]["results"]
    mean3 = mean(res3)
    took3 = mean(results["0.01"]["took_opts"])

    plt.plot(x, res3)
    plt.legend([f"y = e=0.01, mean = {int(mean3)}, exploit_steps={int(took3)}"])
    plt.show()

    print("nearly done")

    x = [i for i in range(1000)]
    avg_revs1 = results["0"]["rewards"]
    plt.plot(x, avg_revs1)
    avg_revs2 = results["0.1"]["rewards"]
    plt.plot(x, avg_revs2)
    avg_revs3 = results["0.01"]["rewards"]
    plt.plot(x, avg_revs3)
    plt.legend([f"e=0 (greedy)", f"e=0.1", f"e=0.01"])
    plt.title("Average Reward")
    plt.show()

def experiment():
    iterations = 5000
    runs_per_game = 1000
    results = {"0": {"results": [],
                     "took_opts": [],
                     "rewards": np.zeros(runs_per_game)},
               "0.1": {"results": [],
                       "took_opts": [],
                       "rewards": np.zeros(runs_per_game)},
               "0.01": {"results": [],
                        "took_opts": [],
                        "rewards": np.zeros(runs_per_game)}
               }
    for i in range(iterations):
        rew, took_opt = run_k_armed_bandit(k=10, e=0, runs=runs_per_game)
        results["0"]["rewards"] += rew
        results["0"]["results"].append(mean(rew))
        results["0"]["took_opts"].append(took_opt)

        rew, took_opt = run_k_armed_bandit(k=10, e=0.1, runs=runs_per_game)
        results["0.1"]["rewards"] += rew
        results["0.1"]["results"].append(mean(rew))
        results["0.1"]["took_opts"].append(took_opt)

        rew, took_opt = run_k_armed_bandit(k=10, e=0.01, runs=runs_per_game)
        results["0.01"]["rewards"] += rew
        results["0.01"]["results"].append(mean(rew))
        results["0.01"]["took_opts"].append(took_opt)

    # avg the results per step
    results["0"]["rewards"] = results["0"]["rewards"] / iterations
    results["0.1"]["rewards"] = results["0.1"]["rewards"] / iterations
    results["0.01"]["rewards"] = results["0.01"]["rewards"] / iterations

    plot_results(results)

if __name__ == "__main__":
    experiment()
