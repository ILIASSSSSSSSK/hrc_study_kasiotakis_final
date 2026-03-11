import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_correct_trait_scores(group_responses, plus_keyed, minus_keyed, trait_items):
    # Initialize trait scores
    trait_scores = {trait: 0 for trait in trait_items}
    num_respondents = len(group_responses)
    num_questions = max(max(plus_keyed), max(minus_keyed))  # Assuming the highest question number

    for responses in group_responses:
        # Ensure responses list matches the number of questions
        if len(responses) != num_questions:
            continue  # Skip this response set if it does not match the expected length

        # Adjusting scores for plus_keyed and minus_keyed items
        adjusted_scores = [0] * num_questions  # Initialize a list of zeroes
        for i in range(1, num_questions + 1):
            if i in plus_keyed and i <= len(responses):
                adjusted_scores[i - 1] = responses[i - 1]
            elif i in minus_keyed and i <= len(responses):
                adjusted_scores[i - 1] = 6 - responses[i - 1]

        # Summing scores for each trait
        for trait, items in trait_items.items():
            valid_items = [item for item in items if item <= num_questions]  # Filter out invalid indices
            trait_score = sum(adjusted_scores[item - 1] for item in valid_items) / len(valid_items)
            trait_scores[trait] += trait_score / num_respondents

    return trait_scores

# Define plus-keyed and minus-keyed items for the Big Five personality test
plus_keyed_big_five = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 31, 33, 35, 37, 40, 41, 42, 43, 45, 47, 48, 50]
minus_keyed_big_five = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29, 30, 32, 34, 36, 38, 39, 44, 46, 49]

# Define the trait-specific item numbers for the Big Five
trait_items_big_five = {
    "Extraversion": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46],
    "Agreeableness": [2, 7, 12, 17, 22, 27, 32, 37, 42, 47],
    "Conscientiousness": [3, 8, 13, 18, 23, 28, 33, 38, 43, 48],
    "Emotional Stability": [4, 9, 14, 19, 24, 29, 34, 39, 44, 49],
    "Intellect/Imagination": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
}






#No transfer group
responsesDXT1=[4,2,2,2,3,2,5,5,2,1,4,1,4,4,4,2,4,4,1,5,5,4,3,2,2,3,3,4,2,1,4,1,4,3,4,4,4,2,2,2,2,4,1,4,4,2,3,5,5,4]

responsesAXAXG1= [4,2,3,2,5,4,4,3,4,2,4,1,4,5,4,3,4,3,2,1,3,1,4,4,3,2,4,1,3,2,4,2,5,5,4,3,4,2,4,4,2,4,5,4,3,5,4,4,2,4]

responsesVXC1=[2,2,2,2,2,2,5,1,2,2,3,2,5,5,5,4,4,3,4,3,4,2,5,4,4,2,2,2,4,1,2,2,5,3,3,2,4,1,3,4,4,4,4,4,4,2,4,5,3,5]

responsesDXM1=[2,1,2,4,3,2,5,2,3,1,4,1,5,2,5,2,5,2,2,2,4,1,2,1,4,2,4,2,3,1,2,1,5,1,4,4,4,2,1,1,4,5,4,1,5,4,4,4,3,4]

responsesKXI1=[2,4,3,3,4,1,2,4,4,2,4,3,5,4,5,2,3,2,4,3,4,4,2,3,4,3,4,1,3,1,3,4,4,4,5,3,3,2,4,4,3,3,5,4,3,3,3,4,4,4]

#responsesKXP1=[2,1,4,2,4,3,4,4,2,3,4,3,4,3,3,3,4,2,3,2,4,1,2,2,5,2,1,1,1,2,4,2,4,3,5,3,4,2,1,2,4,3,4,3,3,2,4,4,3,4]####wrong?
responsesKXP1=[4,2,3,2,5,4,4,3,4,2,4,1,4,5,4,3,4,3,2,1,3,1,4,4,3,2,4,1,3,2,4,2,5,5,4,3,4,2,4,4,2,4,5,4,3,5,4,4,2,4]####wrong?

responsesTXS1=[2,4,5,2,4,2,2,1,4,1,4,2,3,4,2,2,1,1,2,1,4,2,1,2,4,1,2,2,1,3,5,4,4,2,3,2,2,1,1,1,4,2,5,1,4,1,3,5,2,4]

responsesFXS1=[3,5,4,2,4,2,5,3,3,2,4,2,4,4,3,3,4,2,3,2,3,1,3,2,3,2,4,2,1,3,2,1,4,1,5,4,4,2,1,3,2,4,3,2,3,3,5,5,3,4]


# Participant responses for the No Transfer group
responses_no_transfer_group = [
    responsesDXT1, responsesAXAXG1, responsesVXC1, responsesDXM1, 
    responsesKXI1, responsesKXP1, responsesTXS1, responsesFXS1
]

# Calculate trait scores for each participant in the No Transfer group
trait_scores_no_transfer_group = [calculate_correct_trait_scores(
    [response], plus_keyed_big_five, minus_keyed_big_five, trait_items_big_five
) for response in responses_no_transfer_group]

# Calculate the mean and standard deviation for each trait across the No Transfer group
mean_scores_no_transfer, std_scores_no_transfer = {}, {}
for trait in trait_items_big_five.keys():
    mean_scores_no_transfer[trait] = np.mean([scores[trait] for scores in trait_scores_no_transfer_group])
    std_scores_no_transfer[trait] = np.std([scores[trait] for scores in trait_scores_no_transfer_group])

# Plotting the radar chart for the No Transfer group
traits = list(mean_scores_no_transfer.keys())
num_traits = len(traits)
angles = np.linspace(0, 2 * np.pi, num_traits, endpoint=False).tolist()
angles += angles[:1]

mean_values = [mean_scores_no_transfer[trait] for trait in traits] + [mean_scores_no_transfer[traits[0]]]
std_dev_values = [std_scores_no_transfer[trait] for trait in traits] + [std_scores_no_transfer[traits[0]]]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
ax.fill(angles, mean_values, 'b', alpha=0.1)  # Fill area with color
ax.plot(angles, mean_values, 'bo-', linewidth=2)  # Plot the mean values
ax.fill_between(angles, np.array(mean_values) - np.array(std_dev_values), np.array(mean_values) + np.array(std_dev_values), color='blue', alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(traits)
plt.title("Big 5 Personality Traits - No Transfer Group (Mean ± Std Dev)")
#plt.show()




##########################################TL GROUP BIG 5########################

responsesMXS1=[2,1,5,4,3,2,5,1,4,1,3,2,5,4,4,2,4,2,3,2,4,2,2,2,3,1,2,2,2,1,4,2,5,2,4,3,3,3,3,4,4,5,4,2,5,5,4,4,3,5]

responsesGXL1=[4,4,2,3,4,3,4,2,2,2,4,2,4,4,2,2,4,1,2,2,3,2,3,2,3,1,4,1,3,2,3,2,4,2,4,2,5,2,2,4,2,5,2,2,5,2,4,3,2,4]

responsesTXM1=[4,4,4,2,4,3,4,3,4,2,4,3,4,4,4,2,4,3,2,2,4,2,3,3,4,2,3,2,2,1,4,1,4,3,4,3,3,2,3,3,3,4,4,4,3,3,4,4,2,4]

responsesIXR1=[4,2,3,3,3,2,4,2,4,3,4,3,3,2,3,3,4,2,4,4,4,2,2,2,3,2,4,2,1,3,3,1,4,1,4,4,3,4,1,3,4,4,3,1,1,3,5,3,1,3]

responsesXXK1=[3,2,4,4,4,3,4,2,4,3,4,2,5,4,4,2,4,1,2,3,4,2,4,3,4,2,2,2,2,2,4,2,5,4,4,2,5,2,2,4,4,4,5,3,4,4,4,5,2,4]

responsesPXG1=[2,3,4,5,2,3,4,4,1,2,4,1,4,5,3,3,4,2,1,1,4,2,2,3,2,4,2,5,4,4,1,1,3,4,3,3,2,1,5,2,4,4,5,2,4,2,2,5,4,3]

responsesKXP1=[2,1,4,2,4,3,4,4,2,3,4,3,4,3,3,3,4,2,3,2,4,1,2,2,5,2,1,1,1,2,4,2,4,3,5,3,4,2,1,2,4,3,4,3,3,2,4,4,3,4]

responsesSTX1=[3,2,3,2,3,2,5,4,4,3,4,2,3,2,4,2,3,2,2,2,5,1,3,2,3,2,2,4,2,2,4,1,4,1,4,3,4,2,2,3,4,3,4,2,4,2,3,3,2,4]
# Participant responses for the TL group
responses_tl_group = [
    responsesMXS1, responsesGXL1, responsesTXM1, responsesIXR1,
    responsesXXK1, responsesPXG1, responsesKXP1, responsesSTX1
]

# Calculate trait scores for each participant in the TL group
trait_scores_tl_group = [calculate_correct_trait_scores(
    [response], plus_keyed_big_five, minus_keyed_big_five, trait_items_big_five
) for response in responses_tl_group]

# Calculate the mean and standard deviation for each trait across the TL group
mean_scores_tl, std_scores_tl = {}, {}
for trait in trait_items_big_five.keys():
    mean_scores_tl[trait] = np.mean([scores[trait] for scores in trait_scores_tl_group])
    std_scores_tl[trait] = np.std([scores[trait] for scores in trait_scores_tl_group])

# Plotting the radar chart for the TL group
mean_values_tl = [mean_scores_tl[trait] for trait in traits] + [mean_scores_tl[traits[0]]]
std_dev_values_tl = [std_scores_tl[trait] for trait in traits] + [std_scores_tl[traits[0]]]

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
ax.fill(angles, mean_values_tl, 'g', alpha=0.1)  # Fill area with color
ax.plot(angles, mean_values_tl, 'go-', linewidth=2)  # Plot the mean values
ax.fill_between(angles, np.array(mean_values_tl) - np.array(std_dev_values_tl), np.array(mean_values_tl) + np.array(std_dev_values_tl), color='green', alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(traits)
plt.title("Big 5 Personality Traits - TL Group (Mean ± Std Dev)")
#plt.show()



# Function to plot the radar chart with intermediate gridlines and a maximum radius of 5
def plot_radar_chart(mean_scores, std_scores, group_name, color):
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    mean_values = [mean_scores[trait] for trait in traits] + [mean_scores[traits[0]]]
    std_dev_values = [std_scores[trait] for trait in traits] + [std_scores[traits[0]]]
    
    ax.fill(angles, mean_values, color, alpha=0.1)
    ax.plot(angles, mean_values, 'o-', color=color, linewidth=2)
    ax.fill_between(angles, np.array(mean_values) - np.array(std_dev_values), np.array(mean_values) + np.array(std_dev_values), color=color, alpha=0.2)

    ax.set_yticks(np.arange(0, 5.5, 0.5))
    ax.set_yticklabels([str(i) for i in np.arange(0, 5.5, 0.5)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(traits)
    plt.title(f"Big 5 Personality Traits - {group_name} Group (Mean ± Std Dev)")
    #plt.show()

# Plotting the radar charts for both groups
plot_radar_chart(mean_scores_no_transfer, std_scores_no_transfer, "No Transfer", 'blue')
plot_radar_chart(mean_scores_tl, std_scores_tl, "TL", 'blue')
plt.show()
