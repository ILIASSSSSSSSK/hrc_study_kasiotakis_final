import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

####################################JOC#########################
#No transfer group
jocresponsesDXT1=[4,6,8,7,8,8]

jocresponsesAXAXG1= [6,7,7,8,8,9]

jocresponsesVXC1=[1,2,3,5,5,5]

jocresponsesDXM1=[4,4,6,7,8,8]

jocresponsesKXI1=[1,2,2,5,6,7]

jocresponsesKXP1=[4,5,6,7,9,9]####wrong?

jocresponsesTXS1=[2,2,3,2,2,3]

jocresponsesFXS1=[5,4,6,7,7,7]



#TL GROUP

jocresponsesMXS1=[6,6,8,8,9,9]

jocresponsesGXL1=[5,7,9,9,9,9]

jocresponsesTXM1=[2,2,2,4,5,4]

jocresponsesIXR1=[7,6,8,8,7,7]

jocresponsesXXK1=[3,5,7,8,8,9]

jocresponsesPXG1=[3,5,6,5,7,8]

jocresponsesKXP1=[4,5,6,7,9,9]####wrong?

jocresponsesSTX1=[4,5,6,6,6,7]



# Calculate means and standard deviations for each group
blocks = ['Baseline', '2', '3', '4', '5', '6']
no_transfer_means = np.mean([jocresponsesDXT1, jocresponsesAXAXG1, jocresponsesVXC1, jocresponsesDXM1, 
                             jocresponsesKXI1, jocresponsesKXP1, jocresponsesTXS1, jocresponsesFXS1], axis=0)
no_transfer_stds = np.std([jocresponsesDXT1, jocresponsesAXAXG1, jocresponsesVXC1, jocresponsesDXM1, 
                           jocresponsesKXI1, jocresponsesKXP1, jocresponsesTXS1, jocresponsesFXS1], axis=0)

TL_means = np.mean([jocresponsesMXS1, jocresponsesGXL1, jocresponsesTXM1, jocresponsesIXR1, 
                    jocresponsesXXK1, jocresponsesPXG1, jocresponsesKXP1, jocresponsesSTX1], axis=0)
TL_stds = np.std([jocresponsesMXS1, jocresponsesGXL1, jocresponsesTXM1, jocresponsesIXR1, 
                  jocresponsesXXK1, jocresponsesPXG1, jocresponsesKXP1, jocresponsesSTX1], axis=0)

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(blocks, no_transfer_means, yerr=no_transfer_stds, fmt='-o', label='No Transfer Group', capsize=5)
plt.errorbar(blocks, TL_means, yerr=TL_stds, fmt='-o', label='TL Group', capsize=5)

plt.title('JoC between the 2 groups')
plt.xlabel('Blocks')
plt.ylabel('Judgment of Control (JoC)')
plt.legend()
plt.grid(True)
#plt.show()



####################################Human Robot collaboration#########################
#No transfer group
jocresponsesDXT1=[4,4,3,2,4,4,4,4,4,2,3,4,4,4,4,4,4,3,3,3]

jocresponsesAXAXG1= [4,5,3,4,2,5,4,5,2,4,2,2,4,3,3,3,3,3,2,2]

jocresponsesVXC1=[4,4,4,4,2,4,4,5,2,4,3,4,4,3,4,4,4,4,4,4]

jocresponsesDXM1=[4,5,5,2,2,4,5,4,3,3,3,2,4,3,4,4,4,4,2,2]

jocresponsesKXI1=[3,4,4,3,3,4,5,4,3,3,4,4,4,4,4,4,4,4,2,2]

jocresponsesKXP1=[3,3,3,3,3,4,5,3,2,2,2,2,3,3,3,4,4,3,3,3] #WRONG CHANGED

jocresponsesTXS1=[4,4,3,4,2,4,5,2,4,1,1,1,1,2,1,4,2,2,4,4]

jocresponsesFXS1=[3,4,4,3,3,4,4,2,4,2,1,2,4,3,3,4,4,3,2,4]



#TL GROUP

#jocresponsesMXS1=[3,3,3,3,3,4,5,3,2,2,2,2,3,3,3,4,4,3,3,3]
jocresponsesMXS11=[5,5,5,4,2,5,5,5,3,3,4,4,4,4,5,4,4,4,2,2]####wrong?

jocresponsesGXL1=[4,4,5,4,2,5,5,5,2,4,3,4,4,3,4,4,3,4,3,2]

jocresponsesTXM1=[3,4,3,3,3,3,3,3,3,3,2,1,3,3,3,3,4,3,3,3]

jocresponsesIXR1=[3,3,3,3,3,3,3,3,3,4,2,2,3,3,4,3,4,3,2,1]

jocresponsesXXK1=[4,5,4,3,3,5,5,5,3,3,4,4,4,4,4,5,4,4,2,2]

jocresponsesPXG1=[4,5,5,4,3,4,5,5,2,4,2,2,4,3,4,3,4,3,2,2]

jocresponsesKXP1=[5,5,5,4,2,5,5,5,3,3,4,4,4,4,5,4,4,4,2,2]####wrong?

jocresponsesSTX1=[3,4,3,3,3,4,4,4,4,2,2,3,3,2,2,2,2,4,2,3]


import numpy as np
import matplotlib.pyplot as plt

# Define the scoring keys for +keyed and -keyed items
plus_keyed = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
minus_keyed = [4, 19, 20]

# Define the trait-specific item numbers and their corresponding keys
trait_items = {
    "Fluency": [1, 2, 3],
    "Contribution": [4, 5],
    "Trust": [11, 12],
    "Teammate traits": [13, 14, 15, 16],
    "Improvement": [6, 7, 8, 9, 10],
    "Alliance": [17, 18, 19, 20]
}

# Function to calculate correct trait scores
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

# Function to plot comparison bar chart
def plot_group_comparison_bar_chart(group1_scores, group2_scores, group1_label, group2_label):
    n_traits = len(group1_scores)
    traits = list(group1_scores.keys())

    index = np.arange(n_traits)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(index, list(group1_scores.values()), bar_width, label=group1_label, alpha=0.6, color='b')
    bars2 = ax.bar(index + bar_width, list(group2_scores.values()), bar_width, label=group2_label, alpha=0.6, color='g')

    ax.set_xlabel('Traits')
    ax.set_ylabel('Scores')
    ax.set_title('Collaboration metrics between the 2 groups')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(traits)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming you have the response data for each group in the following variables:
# jocresponsesDXT1, jocresponsesAXAXG1, ..., jocresponsesSTX1

# Example usage
correct_trait_scores_no_transfer = calculate_correct_trait_scores(
    [jocresponsesDXT1, jocresponsesAXAXG1, jocresponsesVXC1, jocresponsesDXM1,
     jocresponsesKXI1, jocresponsesKXP1, jocresponsesTXS1, jocresponsesFXS1],
    plus_keyed, minus_keyed, trait_items
)

correct_trait_scores_tl = calculate_correct_trait_scores(
    [jocresponsesMXS11, jocresponsesGXL1, jocresponsesTXM1, jocresponsesIXR1,
     jocresponsesXXK1, jocresponsesPXG1, jocresponsesKXP1, jocresponsesSTX1],
    plus_keyed, minus_keyed, trait_items
)
# Plot the bar chart for comparison
plot_group_comparison_bar_chart(correct_trait_scores_no_transfer, correct_trait_scores_tl, "No Transfer Group", "TL Group")
plt.show()