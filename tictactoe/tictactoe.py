"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # 如果棋盘上 X 的数量等于 O 的数量，那么下一步轮到 X（因为 X 总是先手）。
    # 如果棋盘上 X 的数量大于 O 的数量，那么下一步轮到 O。
    x_count = 0
    o_count = 0
    # 遍历棋盘，计算X和O的数量
    for row in board:
        x_count += row.count(X)
        o_count += row.count(O)
    # 判断下一个玩家
    if x_count > o_count:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    # 遍历棋盘每一行和列
    for i in range(len(board)):
        for j in range(len(board[i])):
            # 检查是否是空位
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions

class ActionError(ValueError):
    """Exception raised for errors in the input actions in a board game."""
    # 定义一个类，专门用来指出错误
    def __init__(self, message="Invalid action"):
        super().__init__(message)


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    new_board = copy.deepcopy(board)
    i, j = action
    # 检查行动是否有效（即对应位置是否为空）。这个很傻，因为我觉得只有在人为调用这个函数的时候这个检查才会有效，但是使用runner.py运行时，这个条件永远不会成立，完全是为了过测试而写的。
    if new_board[i][j] != EMPTY:
        raise ActionError("Invalid action: Position already taken.")
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
        raise ActionError("Action is out of bounds")

    # 获取当前行动的玩家
    current_player = player(board)
    # 在新棋盘上放置当前玩家的标记
    new_board[i][j] = current_player
    return new_board



def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # 检查水平和垂直行是否有获胜者
    for i in range(3):
        # 检查水平行
        # 这个写法干净又利落，如果三个相等且不为空
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        # 检查垂直列
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]

    # 检查两个对角线
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]

    # 如果没有获胜者
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # 检查是否有获胜者：使用之前定义的 winner 函数来判断是否有玩家获胜。
    if winner(board) is not None:
        return True
    # 检查棋盘是否已满：遍历棋盘，如果发现有 'EMPTY' 标记，表示棋盘未满，否则棋盘已满。
    for row in board:
        if EMPTY in row:
            return False
    # 如果没有空位且没有获胜者，则游戏结束，平局
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # 使用之前定义的winner函数来确定是否有赢家
    winner_player = winner(board)
    if winner_player == X:
        return 1
    elif winner_player == O:
        return -1
    else:
        # 如果没有赢家，检查是否平局或游戏还在进行中
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # 索性就写两个函数，哪种情况就调用哪个

    def max_value(board):
        if terminal(board):
            return utility(board), None
        v = float('-inf')   # float('-inf') 表示负无穷大
        best_action = None
        for action in actions(board):
            val, _ = min_value(result(board, action))
            if val > v:
                v, best_action = val, action
        return v, best_action

    def min_value(board):
        if terminal(board):
            return utility(board), None
        v = float('inf')
        best_action = None
        for action in actions(board):
            val, _ = max_value(result(board, action))
            if val < v:
                v, best_action = val, action
        return v, best_action

    # 判断当前是哪位玩家的回合
    current_player = player(board)
    # 根据当前玩家是Maximizer还是Minimizer决定使用哪个函数
    if current_player == X:
        _, action = max_value(board)
    else:
        _, action = min_value(board)
    return action

    # 下面是代码的思路
    # Function Max-Value(state):
    # v = -∞
    # if Terminal(state):
    #   return Utility(state)
    # for action in Actions(state):
    #   v = Max(v, Min-Value(Result(state, action)))
    # return v
    # Function Min-Value(state):
    # v = ∞
    # if Terminal(state):
    #   return Utility(state)
    # for action in Actions(state):
    #   v = Min(v, Max-Value(Result(state, action)))
    # return v

