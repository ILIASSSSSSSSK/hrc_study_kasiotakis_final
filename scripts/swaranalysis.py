import numpy as np
import matplotlib.pyplot as plt

# Define the responses as a list
#No transfer group
responsesDXT1=[2,2,1,4,1,3,6,1,2,3,1,1,3,5,5,5,3,1,2,3,2]
responsesAXAXG1= [3,3,2,2,2,4,2,2,2,2,2,3,2,2,4,4,1,2,3,5,2]

responsesVXC1=[2,3,1,2,2,2,5,1,2,2,1,3,1,1,2,1,1,2,1,3,2]

responsesDXM1=[2,4,1,5,4,2,2,1,1,3,1,1,3,2,2,2,3,1,1,2,3]

responsesKXI1=[3,3,2,2,3,1,3,2,3,2,1,3,1,3,2,2,2,1,1,5,2]

responsesKXP1=[2,4,1,3,2,3,3,1,2,2,2,2,2,1,2,3,2,1,2,3,3]

responsesTXS1=[4,2,1,2,1,3,1,5,5,4,5,6,1,2,4,1,1,1,2,3,5]

responsesFXS1=[2,5,1,5,2,4,4,2,1,2,1,1,2,1,3,3,4,1,2,3,3]


#TL group
responsesMXS1=[1,2,4,3,3,3,3,2,4,3,2,3,5,4,1,3,3,3,1,4,3]

responsesGXL1=[4,5,2,4,5,3,4,6,5,5,6,6,5,5,3,5,5,5,1,2,5]

responsesTXM1=[3,4,4,3,4,2,4,3,4,2,2,3,4,4,2,3,2,3,3,3,2]

responsesIXR1=[5,5,1,3,4,3,3,2,2,2,2,1,2,1,3,2,2,2,2,3,2]

responsesXXK1=[2,4,2,2,2,2,2,1,2,2,2,2,2,2,2,3,3,1,2,4,3]

responsesPXG1=[3,5,1,2,5,3,3,3,4,5,6,3,3,5,2,2,5,2,3,5,6]

responsesKXP1=[2,4,1,3,2,3,3,1,2,2,2,2,2,1,2,3,2,1,2,3,3]

responsesSTX1=[3,5,2,2,5,4,1,3,4,5,1,2,3,2,1,2,2,3,1,2,4]




# Define the responses for each group
group1_responses = [responsesDXT1, responsesAXAXG1, responsesVXC1, responsesDXM1, responsesKXI1, responsesKXP1, responsesTXS1, responsesFXS1]
group2_responses = [responsesMXS1, responsesGXL1, responsesTXM1, responsesIXR1, responsesXXK1, responsesPXG1, responsesKXP1, responsesSTX1]

# Define the characteristics and the number of questions for each
characteristics = {
    "Benevolence": 2,
    "Universalism": 3,
    "Self-Direction": 2,
    "Stimulation": 2,
    "Hedonism": 2,
    "Achievement": 2,
    "Power": 2,
    "Security": 2,
    "Conformity": 2,
    "Tradition": 2
}

# Create a function to calculate mean and std for each characteristic
def calculate_mean_std(responses):
    scores = {}
    start_idx = 0

    for characteristic, num_questions in characteristics.items():
        end_idx = start_idx + num_questions
        # Calculate mean for each question within a trait for each participant
        characteristic_means = responses[:, start_idx:end_idx].mean(axis=1)
        # Calculate overall mean and standard deviation for each trait across all participants
        scores[characteristic] = (characteristic_means.mean(), characteristic_means.std())
        start_idx = end_idx

    return scores

# Calculate mean and std for each group
group1_scores = calculate_mean_std(np.array(group1_responses))
group2_scores = calculate_mean_std(np.array(group2_responses))

# Create spider web plots for each group
def create_spider_web_plot(scores, group_name, color):
    # Extract categories and their corresponding mean and std deviation values
    categories = list(scores.keys())
    means = [score[0] for score in scores.values()]
    stds = [score[1] for score in scores.values()]

    # Determine the number of categories
    num_categories = len(categories)

    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    # Extend means and stds to close the plot
    means += means[:1]
    stds += stds[:1]

    # Create spider plot
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))

    # Plot data and fill area
    ax.fill(angles, means, color, alpha=0.1)
    ax.plot(angles, means, 'o-', color=color, linewidth=2)
    ax.fill_between(angles, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.2)

    # Set the y-axis tick marks and labels
    ax.set_yticks(np.arange(0, 5.5, 0.5))
    ax.set_yticklabels([str(i) for i in np.arange(0, 5.5, 0.5)])

    # Set the trait labels
    ax.set_xticks(angles[:-1])
    
    # Rotate the category labels and values one position to the left (anti-clockwise)
    rotated_categories = categories[-1:] + categories[:-1]
    rotated_means = means[-1:] + means[:-1]
    rotated_stds = stds[-1:] + stds[:-1]

    ax.set_xticklabels(rotated_categories)

    # Set the title
    plt.title(f"{group_name} Traits (Mean ± Std Dev)")

    plt.show()




# Create spider web plots for both groups
create_spider_web_plot(group1_scores, "No TL Group", "blue")
create_spider_web_plot(group2_scores, "TL group", "blue")
plt.show()
