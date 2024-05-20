import math
import random
import time

# 原理就是获得每一步状态的价值，从而确定走不同步后，得到的局面的价值，这个是马尔可夫决策过程
# 下面是马尔可夫决策过程的几个因素
# 行动空间（Action Space）：定义了在给定状态下可以采取的所有可能行动。
# 转移概率（Transition Probabilities）：描述了在给定当前状态和行动下，转移到下一个状态的概率。
# 奖励函数（Reward Function）：给出了在转移到新状态时所获得的即时奖励。
# 折扣因子（Discount Factor）：用来调节未来奖励相对于即时奖励的重要性。
# 那么在这里，怎么确定我们要的局面价值呢？就通过强化学习，从而获得所有局面的价值，在运行过程中，是从后面推到前面来获得局面价值
# Q(s, a) ⟵ Q(s, a) + α(r+γ*maxQ(s',a') - Q(s, a))

class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        初始化游戏板。
        每个游戏板包含：
            - `piles`: 一个列表，表示每一堆中剩余元素的数量
            - `player`: 0 或 1，表示轮到哪位玩家
            - `winner`: None, 0, 或 1，表示赢家是谁
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) 输入一个 `piles` 列表，
        返回该状态下所有可用的动作 `(i, j)`。

        动作 `(i, j)` 表示从第 i 堆移除 j 个元素（堆是从 0 开始编号的）。
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions  # 返回所有可能的动作集合

    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1  # 如果当前玩家是1，则返回0，反之返回1

    def switch_player(self):
        """
        切换当前玩家。
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        对当前玩家执行动作 `action`。
        `action` 必须是一个元组 `(i, j)`。
        """
        pile, count = action

        # 先判断移除堆的行为是否合法
        if self.winner is not None:
            raise Exception("Game already won")  # 如果游戏已经结束，则抛出异常
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")     # 如果堆索引不合法，抛出异常
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")    # 如果移除的数量不合法，抛出异常

        # 更新堆
        self.piles[pile] -= count
        self.switch_player()

        # C检查是否有赢家
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        用空的 Q-学习字典初始化 AI，设置 alpha（学习率）和 epsilon（探索率）。

        Q-学习字典将 `(state, action)` 对映射到一个 Q-值（一个数值）。
         - `state` 是剩余堆的元组，例如 (1, 1, 4, 4)
         - `action` 是一个行动的元组 `(i, j)`
        """
        # 下面这个dict就是我们所谓的AI，里面放了所有的步骤
        self.q = dict() # 初始化一个空字典用于存储 Q-值
        self.alpha = alpha   # 设置学习率
        self.epsilon = epsilon # 设置探索率

    def update(self, old_state, action, new_state, reward):
        """
        个人人为，这里应该不叫new_state，应该叫下一个局面
        根据旧状态、在该状态下采取的行动、新的结果状态以及从采取该行动获得的奖励来更新 Q-学习模型。
        """
        old = self.get_q_value(old_state, action)   # 获取旧的 Q-值
        best_future = self.best_future_reward(new_state)    # 计算最佳未来奖励
        self.update_q_value(old_state, action, old, reward, best_future)    # 更新 Q-值

    def get_q_value(self, state, action):
        """
        返回给定状态 `state` 和行动 `action` 的 Q-值。
        如果在 `self.q` 中还没有 Q-值，则返回 0。
        """
        return self.q.get((tuple(state), tuple(action)), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        根据先前的 Q-值 `old_q`、当前奖励 `reward` 和对未来奖励的估计 `future_rewards`，
        更新状态 `state` 和行动 `action` 的 Q-值。

        Use the formula:
        Q(s, a) <- 旧价值估计 + alpha * (新价值估计 - 旧价值估计)

        其中 `旧价值估计` 是之前的 Q-值，
        `alpha` 是学习率，alpha的变动可以从旧价值估计滑动到新价值估计
        `新价值估计` 是当前奖励和预估未来奖励的总和。
        """
        # 计算新的价值估计，这里我觉得可以调整一下，不一定是要加法，可以带一点乘法
        # future_rewards是下一个局面的价值
        new_value_estimate = reward + future_rewards
        # 更新 Q 值
        new_q = old_q + self.alpha * (new_value_estimate - old_q)
        # 将更新后的 Q 值存储回 Q-值字典
        self.q[(tuple(state), tuple(action))] = new_q


    def best_future_reward(self, state):
        """
        给定一个状态 `state`，考虑该状态下所有可能的 `(state, action)` 对，
        返回所有这些对的 Q-值的最大值。

        如果一个 `(state, action)` 对在 `self.q` 中没有 Q-值，则使用 0 作为 Q-值。
        如果在 `state` 中没有可用的行动，则返回 0。
        """
        # 下面是一个小知识
        # available_actions 是另一个类中的静态方法或类方法，可以像下面那样使用它。
        possible_actions = Nim.available_actions(state)

        # 检查是否有可用行动
        if not possible_actions:
            return 0  # 如果没有可用行动，则直接返回 0
        # 上面我有点看法，当没有可用行动时返回一个负值可能更加合适，因为这可以反映出这种状态的不利程度。在游戏如 Nim 中，如果一方无法行动，则通常意味着该方已经输了，因此给予一个负值作为奖励可以更准确地反映出这种状态的不利性。

        # 对于每个可能的行动，获取 Q 值，如果不存在则为 0
        max_reward = 0  # 初始化最大奖励
        for action in possible_actions:
            # 获取 Q 值，如果 `(state, action)` 对没有 Q 值，则默认为 0
            q_value = self.q.get((tuple(state), tuple(action)), 0)
            # 更新最大奖励
            if q_value > max_reward:
                max_reward = q_value

        # 返回所有可能行动的最大 Q 值
        return max_reward


    def choose_action(self, state, epsilon=True):
        """
        给定一个状态 `state`，返回要采取的行动 `(i, j)`。
        如果 `epsilon` 为 `False`，则返回该状态下可用的最佳行动（Q-值最高的行动，
        对于没有 Q-值的对使用 0）。
        如果 `epsilon` 为 `True`，则以概率 `self.epsilon` 随机选择一个可用的行动，
        否则选择最佳可用行动。
        如果多个行动有相同的 Q-值，任何这些选项都是可接受的返回值。
        """
        # 一种是始终选择最佳行动（即拥有最高Q - 值的行动），另一种是利用epsilon贪婪策略，即以一定概率随机选择行动，以促进探索。这个函数的实现将依赖于能够获取给定状态下所有可能行动的功能，以及计算每个行动的Q - 值。
        possible_actions = Nim.available_actions(state)  # 假设有函数返回所有可能的行动
        if not possible_actions:
            return None  # 如果没有可行的行动，返回 None

        if not epsilon:
            # 选择最佳行动
            best_action = max(possible_actions, key=lambda action: self.q.get((tuple(state), tuple(action)), 0))
            return best_action
        else:
            # 使用 epsilon 贪婪策略
            if random.random() < self.epsilon:
                # 以概率 epsilon 随机选择行动
                return random.choice(list(possible_actions))
            else:
                # 选择最佳行动
                best_action = max(possible_actions, key=lambda action: self.q.get((tuple(state), tuple(action)), 0))
                return best_action

def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = NimAI()

    # Play n games，训练n轮
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            # 这里的作用是记录两个选手中，其中一个选手的动作和局面，以便进行学习
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            # 将ai做的决定输入到game中
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            # 如果分出胜负，可以结束了，并且开始强化，这里是一局结束后才强化，这里这个选手是输掉的，因为他做了最后一步，所以他的步骤要记作-1，另一个选手的步骤要记作1
            if game.winner is not None:
                # 这里我觉得有点小疑惑，这样收敛的速度会不会太慢了，因为-1和+1会抵消，相当于最差局面的分数为0，和未选择的局面分数一样，需要随机性才能破局，如果有更多的时间，我希望自己调调强化参数，我觉得应该将reward调的特别大
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break
            # 如果未分出胜负，奖励为0，让它的当前局面分数完全根据下一个局面
            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")

    # Return the trained AI
    return player


def play(ai, human_player=None):
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether
    human player moves first or second.
    """

    # If no player order set, choose human's order randomly
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # Game loop
    while True:

        # Print contents of piles
        print()
        print("Piles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")
        print()

        # Compute available actions
        available_actions = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let human make a move
        if game.player == human_player:
            print("Your Turn")
            while True:
                pile = int(input("Choose Pile: "))
                count = int(input("Choose Count: "))
                if (pile, count) in available_actions:
                    break
                print("Invalid move, try again.")

        # Have AI make a move
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print()
            print("GAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return


# 关键公式：每次我们处于状态 s 并采取操作 a 时，我们都可以根据以下公式更新 Q 值 Q(s, a) ：
# Q(s, a) <- Q(s, a) + alpha * (new value estimate - old value estimate)
# old value estimate是 Q(s, a) 的现有值,new value estimate 表示当前操作收到的奖励与玩家将收到的所有未来奖励的估计之和
'''
Q-学习是强化学习的一种模型，其中一个函数 Q(s, a) 输出在状态 s 下采取行动 a 的价值估计。
这个模型从所有估计值等于0开始（对所有 s, a，Q(s,a) = 0）。当采取一个行动并获得一个奖励时，函数做两件事：1）基于当前奖励和预期的未来奖励估计 Q(s, a) 的价值；2）更新 Q(s, a) 以同时考虑旧估计和新估计。这提供了一个能够在不从头开始的情况下改进其过去知识的算法。
Q(s, a) ⟵ Q(s, a) + α(新价值估计 - Q(s, a))
Q(s, a) 的更新值等于之前的 Q(s, a) 的值加上一些更新值。这个值由新旧价值之差决定，乘以 α，一个学习系数。当 α = 1 时，新估计简单地覆盖旧的。当 α = 0 时，估计值永远不更新。通过提高和降低 α，我们可以确定以多快的速度用新估计更新以前的知识。
新的价值估计可以表示为奖励（r）和未来奖励估计的总和。为了获得未来奖励估计，我们考虑采取最后一个行动后获得的新状态，并加上在这个新状态中将带来最高奖励的行动的估计。这样，我们不仅通过它收到的奖励，而且通过下一步的预期效用来估计在状态 s 进行行动 a 的效用。未来奖励估计的值有时会出现一个控制未来奖励被重视程度的系数 gamma。最终我们得到以下方程：
Q(s, a) ⟵ Q(s, a) + α(r+γ*maxQ(s',a') - Q(s, a))
r 是立即奖励。α 是学习率，控制新信息覆盖旧信息的速度。γ 是折扣因子，它决定未来奖励相对于即时奖励的重要性。如果γ接近 1，表示未来奖励几乎和即时奖励一样重要；如果γ接近 0，表示几乎只关注即时奖励。
贪婪决策算法完全不考虑未来的估计奖励，而总是选择在当前状态 s 下具有最高 Q(s, a) 的行动 a。
这引出了探索与开发的权衡讨论。一个贪婪算法总是进行开发，采取已知能带来好结果的行动。然而，它总是按照相同的路径解决问题，永远不会找到更好的路径。另一方面，探索意味着算法可能会在前往目标的途中使用之前未探索的路线，从而发现更有效的解决方案。例如，如果你每次都听同样的歌曲，你知道你会享受它们，但你永远不会知道你可能会更喜欢的新歌！
为了实现探索和开发的概念，我们可以使用 ε (epsilon) 贪婪算法。在这种算法中，我们设置 ε 等于我们想要随机移动的频率。以 1-ε 的概率，算法选择最佳移动（开发）。以 ε 的概率，算法选择一个随机移动（探索）。
另一种训练强化学习模型的方式是在整个过程结束时而不是每次移动时给予反馈。例如，考虑一场尼姆游戏。在这个游戏中，不同数量的物体分布在几堆中。每个玩家可以从任何一堆中取走任意数量的物体，最后一个取走物体的玩家输掉游戏。在这样的游戏中，一个未训练的人工智能将随机玩，很容易被击败。为了训练人工智能，它将从随机玩游戏开始，在最后获得赢得1分，输掉-1分的奖励。例如，当它在10,000场游戏上受训后，它已经足够聪明，很难被打败。
当游戏具有多种状态和可能的行动时，例如国际象棋，这种方法变得更加计算要求高。在这种情况下，我们可以使用函数逼近，这允许我们使用各种其他特征来近似 Q(s, a)，而不是为每个状态-行动对存储一个值。因此，算法能够识别哪些移动足够相似，以致它们的估计值应该相似，并在其决策中使用这种启发式。
这个更新过程意味着每次决策不仅考虑即时的结果（立即奖励），还考虑长远的影响（未来奖励的最大估计）。这样，Q-学习算法能够逐步学习到在不同状态下哪些行动能够最大化整体收益，最终使策略趋于最优。
'''