# Speaking in Chess: Training Large Language Models to Play Chess
Author: Evan Frick, Tianle Li, Kayla Lee

Check out the paper! [Paper link](paper.pdf)

### Abstract 
In this paper, we delve into the uncharted territory of chess analytics with Large Language Models (LLMs),
transforming the game’s representation from static positions to a sequence-to-sequence problem. Leveraging a
custom chess move tokenizer, our GPT-2-Small model intakes a series of chess moves and output the next optimal
move by learning exclusively from actions and sparsely distributed rewards of a game’s outcome. After extensive
behavior cloning on over 20 milion high-ELO chess games, this behavior cloning strategy autoregressively
generates full-length chess games with reasonable accuracy.
To further enhance the model, we evaluated 6 Reinforcement Learning (RL) strategies, including 3 novel algo-
rithms, Fictitious Self-Play with Short-Term Adversaries, Past-Present Q-Iteration with a Pseudo-Ensemble, and
Self-Play with Funnel Searching.

First, we attempted self-play through Policy Gradient and Q-Iteration methods. Although elegant, we found that
pure self-play with a single model was too unstable for developing a reliable policy, as the training often deviated
towards draws with low precision game-play.

To improve training stability, we implemented Fictitious Self-Play with Past-Present Q-Iteration, yielding a
dependable policy. Notably, extensive training iterations demonstrated that the present policy consistently outper-
formed the past, indicating optimized reward capture. To ensure the current model’s superiority was not limited to
an adversarial role, we introduced an LLM-based-pseudo-ensemble algorithm. By assigning models different
chess openings at each training step, we generated diverse policy conditionals. However, after numerous iterations,
the present policy struggled to consistently outpace the past model within this pseudo-ensemble framework,
underscoring the complexity of dynamic opening play.

Acknowledging the learning challenges associated with ensemble-based algorithms, we introduced Fictitious
Self-Play with Short-Term Adversaries. This algorithm involves training a base model as an adversary, which is
subsequently employed to refine the current model’s defenses. The adversary is reset each iteration to prevent
over-specialization and the defending policy is given few training steps to prevent reverse adversarial strategies.
We find that Fictitious Self-Play with Short-Term Adversaries is comparable to Past-Present Q-Iteration. Notably,
both the adversary policy and the actual policy trained yield strong and improved policies, even though the
adversary policy is reset to the base model after each training iteration.

Observing the training inefficiencies and slow inference speed, we attempted a simple offline training scheme
where we employ trajectory loss. We find loss converges, but the policy extensively plays out of distribution
moves at test-time.

Finally, our evaluations against the chess engine Stockfish revealed early-game proficiency but mid-game vul-
nerabilities leading to suboptimal late-game play. We developed Self-Play with Funnel Searching, a strategic
exploration method that selectively expands potential moves based on their trajectory probabilities. Funnel Search
samples k moves at a given state and pick top m moves from all sampled moves using a customizable k and m
scheduler, resulting in “funnel” shaped trajectories trees.

Ultimately, we find that pure single-model self-play is too unstable, even with improve game sampling methods
like funnel search. We also find that simple offline methods are not sufficient and are unable to punish out of
distribution strategies. We find the dual-model methods protect against divergence, and promote better policies.
Past-Present Q-iteration and Fictitious Self-Play with Short-Term Adversaries are able to produce policies better
than the behavior cloned base model. Additionally, we find all our implicitly Q-Value altering methods do not
yield useful Q-Values after RL training, which prevents the use of true dynamic programming based algorithms.

We conclude that the sparse-reward action history inferred state learning problem is exceedingly difficult. Reward
signals are noisy as they are passed down to actions earlier in the trajectory. Moreover, the state space exponentially
increases for each action. As such, training successful policies in this spaced is difficult, and requires myriad
methods to overcome the inherent instability of the learning problem. We hope that the novel algorithms present
can be applicable to more than just chess, and can contribute to a wider range of sparse reward sequence problems.
