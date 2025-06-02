import matplotlib.pyplot as plt

def play_vs_other_agent(env, agent1, agent2, render=False, verbose=False):
    """
    Play a game between two agents in the given environment and optionally record it.
    
    Parameters:
        env: The game environment (can be wrapped with VideoRecorderWrapper).
        agent1: The first agent to play.
        agent2: The second agent to play.
        render: Whether to render the game (default: False). Used to record if wrapped.
        verbose: Whether to print detailed game information (default: False).
    
    Returns:
        winner (int): 1 or 2 depending on who won
    """
    obs, info = env.reset()
    done = False

    if render and hasattr(env, "render"):
        env.render()

    while not done:
        if verbose:
            print(f"Current Player: {obs['current_player'] + 1}")

        if obs["current_player"] == 0:
            action = agent1.act(obs)
        else:
            action = agent2.act(obs)

        obs, reward, done, truncated, info = env.step(action)

        if render and hasattr(env, "render"):
            env.render()

        if verbose:
            print(f"Action taken: {action}")
            print(f"Reward received: {reward}")

    if render and hasattr(env, "render"):
        env.render()

    last_player = 1 - obs["current_player"]

    if hasattr(env, 'env') and hasattr(env.env, 'misere'):
        misere = env.env.misere
    else:
        misere = env.misere if hasattr(env, 'misere') else False
    winner = obs["current_player"] if misere else last_player

    if winner == 0:
        print("Agent 1 wins!")
        return 1
    else:
        print("Agent 2 wins!")
        return 2
    
def run_multiple_games(env, agent1, agent2, num_games=100):
    results = {
        "agent1_wins": 0,
        "agent2_wins": 0
    }

    for _ in range(num_games):
        obs, _ = env.reset()
        done = False

        while not done:
            current_agent = agent1 if obs["current_player"] == 0 else agent2
            action = current_agent.act(obs)
            obs, reward, done, _, _= env.step(action)

        # After game ends, the current_player is the one who WOULD play next.
        last_player = 1 - obs["current_player"]

        if hasattr(env, 'env') and hasattr(env.env, 'misere'):
            misere = env.env.misere
        else:
            misere = env.misere if hasattr(env, 'misere') else False
        if env.misere:
            # Last move loses → winner is the one who didn't move last
            winner = obs["current_player"]
        else:
            # Last move wins → winner is the one who moved last
            winner = last_player

        if winner == 0:
            results["agent1_wins"] += 1
        else:
            results["agent2_wins"] += 1

    return results

def plot_results(results):
    # Plotting
    labels = ['Agent 1', 'Agent 2']
    counts = [results["agent1_wins"], results["agent2_wins"]]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, counts, color=['skyblue', 'salmon'])
    plt.ylabel('Wins')
    plt.title('TacTix Tournament Results')

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{int(height)}', ha='center')

    plt.ylim(0, max(counts) + 10)
    plt.show()