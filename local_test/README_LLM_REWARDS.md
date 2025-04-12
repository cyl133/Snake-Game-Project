# LLM-Guided Reward Shaping for Snake RL

This system uses an LLM (Language Learning Model) to dynamically analyze agent behavior and adapt the reward function in a Snake reinforcement learning environment. The system collects detailed metrics, periodically sends them to an LLM to analyze, and adjusts reward values based on LLM recommendations.

## How It Works

1. **Metrics Collection**
   - The system collects detailed metrics for each training episode:
     - Episode length
     - Food collected
     - Death causes (wall/self/timeout)
     - Map coverage
     - Looping behavior
     - Action entropy
     - Turn frequency
     - And many more

2. **LLM-Based Reward Shaping**
   - After every N episodes (default: 500), metrics are sent to the LLM
   - The LLM analyzes agent behavior and suggests reward function modifications
   - The reward function is updated according to the LLM's recommendations

3. **Dynamic Rewards**
   - Instead of fixed rewards, the agent receives dynamic rewards based on:
     - Basic outcomes (food eaten, death)
     - Exploration behavior (visiting unique cells)
     - Looping detection (penalizing repetitive patterns)
     - Position-based rewards (center bonuses, wall penalties)

4. **Logging and Visualization**
   - All metrics and reward modifications are logged to Weights & Biases
   - Evolution of the reward function is tracked and can be visualized
   - Performance metrics can be correlated with reward changes

## Files and Components

- `llm_reward_shaper.py`: Core metrics collection and LLM integration
- `gym_env.py`: Modified Snake environment to use dynamic rewards
- `training_test.py`: Training script with LLM reward integration
- `eval_llm_rewards.py`: Evaluation and visualization script

## Setup and Usage

### Prerequisites

1. Install required packages:
   ```bash
   pip install stable-baselines3 gymnasium wandb matplotlib numpy opencv-python
   ```

2. Set up Weights & Biases:
   ```bash
   wandb login
   ```

3. Set your OpenAI API key (or other LLM service) as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### Training

Run the training script:
```bash
cd local_test
python training_test.py
```

This will:
- Initialize a wandb run for tracking metrics
- Create multiple Snake environments using SubprocVecEnv
- Train the agent using PPO
- Periodically call the LLM to update the reward function
- Log all metrics and reward changes to wandb

### Evaluation

After training, evaluate the agent:
```bash
python eval_llm_rewards.py
```

This will:
1. Plot the evolution of the reward function over time
2. Run evaluation episodes with the trained agent
3. Generate visualizations of the agent's performance

## Configuration

Key parameters you can modify:

- `LLM_CALL_FREQUENCY` in `llm_reward_shaper.py`: How often to call the LLM (in episodes)
- `LLM_MODEL` in `llm_reward_shaper.py`: Which LLM model to use
- Reward components in `MetricsCollector.current_reward_config`: Initial reward values
- Game parameters in `param_configs/eval.json`: Settings for the Snake environment

## Understanding Reward Evolution

The system tracks how rewards change over time. In the evaluation script, you can see:

1. How reward components evolved with LLM suggestions
2. How these changes affected agent performance metrics
3. Whether the agent learned to collect more food, live longer, or explore better

## Example Reward Components

- `food_reward`: Reward for eating food
- `death_penalty`: Penalty for dying
- `step_penalty`: Small penalty per step to encourage efficiency
- `center_bonus`: Reward for visiting center regions (encouraging exploration)
- `loop_penalty`: Penalty for repetitive movement patterns
- `wall_follow_penalty`: Penalty for hugging walls
- `exploration_bonus`: Reward for visiting new cells

The LLM can adjust these values based on observed behavior. For example, if the agent is getting stuck in loops, the LLM might increase the `loop_penalty`.

## Further Customization

You can extend this system by:
1. Adding more metrics to track in `MetricsCollector`
2. Creating additional reward components
3. Modifying the LLM prompt to focus on specific aspects of agent behavior
4. Implementing more sophisticated looping or behavior detection 