from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .q_interactive_learner import QInteractiveLearner
from .q_influence_learner import QInfluenceLearner
from .q_explore_learner import QExploreLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["q_interactive_learner"] = QInteractiveLearner
REGISTRY["q_influence_learner"] = QInfluenceLearner
REGISTRY["q_explore_learner"] = QExploreLearner
