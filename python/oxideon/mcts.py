"""Monte Carlo Tree Search (MCTS) for Reasoning LLMs and Planning Systems.

Implements AlphaGo / DeepSeek-R1 / Q* style tree search for chain-of-thought exploration:
- MCTSNode: State-action node tracking visits, cumulative value, and prior probabilities
- MCTSConfig: Search hyperparameters (c_puct exploration constant, simulations, rollout depth)
- MCTS: Full search loop (UCT selection, expansion, value evaluation, and backpropagation)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import numpy as np


class MCTSConfig:
    """Hyperparameters for Monte Carlo Tree Search."""

    def __init__(
        self,
        num_simulations: int = 50,
        c_puct: float = 1.414,
        max_depth: int = 12,
        discount: float = 1.0,
        temperature: float = 1.0,
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.discount = discount
        self.temperature = temperature


class MCTSNode:
    """A node in the Monte Carlo Tree Search graph."""

    def __init__(
        self,
        state: Any,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Any] = None,
        prior: float = 1.0,
    ):
        self.state = state
        self.parent: Optional["MCTSNode"] = parent
        self.action = action
        self.prior = float(prior)
        self.children: Dict[Any, "MCTSNode"] = {}
        self.visits: int = 0
        self.value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def uct_score(self, c_puct: float = 1.414) -> float:
        """Computes Upper Confidence Bound for Trees (UCT / PUCT) score."""
        parent_visits = self.parent.visits if self.parent is not None else 1
        exploration = c_puct * self.prior * math.sqrt(max(1, parent_visits)) / (1.0 + self.visits)
        return self.q_value + exploration


class MCTS:
    """Monte Carlo Tree Search engine for LLM thought planning and game search."""

    def __init__(self, config: Optional[MCTSConfig] = None):
        self.config = config if config is not None else MCTSConfig()

    def search(
        self,
        initial_state: Any,
        policy_fn: Callable[[Any], List[Tuple[Any, float]]],
        value_fn: Callable[[Any], float],
        is_terminal_fn: Callable[[Any], bool],
        transition_fn: Optional[Callable[[Any, Any], Any]] = None,
    ) -> MCTSNode:
        """Executes MCTS simulations from the root state."""
        root = MCTSNode(state=initial_state)

        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # 1. Selection: descend tree using UCT score
            depth = 0
            while node.is_expanded and not is_terminal_fn(node.state) and depth < self.config.max_depth:
                best_action, best_child = max(
                    node.children.items(),
                    key=lambda item: item[1].uct_score(self.config.c_puct),
                )
                node = best_child
                search_path.append(node)
                depth += 1

            # 2. Expansion: expand candidate actions if not terminal
            if not is_terminal_fn(node.state) and depth < self.config.max_depth:
                candidates = policy_fn(node.state)
                for action, prior in candidates:
                    if transition_fn is not None:
                        next_state = transition_fn(node.state, action)
                    elif isinstance(node.state, list):
                        next_state = node.state + [action]
                    else:
                        next_state = (node.state, action)
                    child_node = MCTSNode(state=next_state, parent=node, action=action, prior=prior)
                    node.children[action] = child_node

            # 3. Evaluation: compute state value estimate
            value = float(value_fn(node.state))

            # 4. Backpropagation: propagate value up the tree
            for n in reversed(search_path):
                n.visits += 1
                n.value_sum += value
                value *= self.config.discount

        return root

    def best_action(self, root: MCTSNode) -> Any:
        """Returns the action corresponding to the most visited child, breaking ties by Q-value."""
        if not root.children:
            return None
        return max(
            root.children.items(),
            key=lambda item: (item[1].visits, item[1].q_value),
        )[0]

    def best_trajectory(self, root: MCTSNode) -> List[Any]:
        """Extracts the greedy trajectory of actions from root to leaf following best_action."""
        actions: List[Any] = []
        curr = root
        while curr.children:
            act = self.best_action(curr)
            if act is None or act not in curr.children:
                break
            actions.append(act)
            curr = curr.children[act]
        return actions

    def action_probabilities(self, root: MCTSNode, temperature: Optional[float] = None) -> Dict[Any, float]:
        """Returns normalized probability distribution across root actions based on visit counts."""
        if not root.children:
            return {}
        temp = self.config.temperature if temperature is None else float(temperature)
        if temp < 0:
            raise ValueError(f"temperature must be non-negative, got {temp}")
        if temp == 0.0:
            best_act = self.best_action(root)
            return {action: (1.0 if action == best_act else 0.0) for action in root.children}

        visits = np.array([float(child.visits) for child in root.children.values()], dtype=np.float64)
        if visits.sum() == 0:
            return {action: 1.0 / len(root.children) for action in root.children}
        if temp != 1.0:
            with np.errstate(divide="ignore"):
                log_v = np.where(visits > 0, np.log(visits) / temp, -np.inf)
            max_log = np.max(log_v)
            exp_v = np.where(visits > 0, np.exp(log_v - max_log), 0.0)
            probs = exp_v / exp_v.sum()
        else:
            probs = visits / visits.sum()

        return {action: float(p) for action, p in zip(root.children.keys(), probs)}


__all__ = ["MCTSConfig", "MCTSNode", "MCTS"]
