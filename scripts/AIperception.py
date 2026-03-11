

# Define the participant's responses as a list of numbers
#participant_responses = [3,4,3,3,4,4,4,2,1,2,4,4,4,4,2,3,4,4,2,3]

#print(len(participant_responses))
# Define the categorization of items as "Pos" (Positive) and "Neg" (Negative)
item_categorization = [
    "Pos", "Pos", "Neg", "Pos",
    "Pos", "Neg", "Pos", "Neg",
    "Neg", "Neg", "Pos", "Pos",

    "Pos", "Pos", "Neg", "Pos",
    "Pos", "Pos", "Neg", "Neg"
]


# Define responses for two groups (8 participants in each group)
group1_responses = [
    [3,4,3,3,4,4,4,2,1,2,4,4,4,4,2,3,4,4,2,3], #DXT
    [2,5,4,3,5,3,5,2,2,3,4,5,4,4,3,3,4,4,2,2], #AXAG
    [1,4,4,4,5,3,4,1,1,3,4,4,4,4,4,3,4,2,1,1], #VXC
    [3,4,3,3,4,4,4,3,2,2,4,3,3,4,3,3,3,4,2,3], #DXM
    [5,5,2,4,5,3,5,2,3,3,4,5,4,4,1,4,4,5,2,2], #KXI
    #[3,4,3,4,4,2,4,2,2,2,4,4,4,4,2,3,4,4,2,3] #KXP###
    [2,3,4,4,5,4,3,2,4,3,4,5,5,4,4,4,3,3,2,4], #TXS
    [3,4,3,3,5,2,4,3,4,3,5,5,4,5,3,2,4,2,2,3] #FXS
]

group2_responses = [
    [3,3,3,1,4,3,3,3,3,3,4,4,4,4,4,3,4,4,2,4], #MXS
    [4,5,3,4,5,3,4,2,1,2,5,5,5,5,2,2,5,5,1,4], #GXL
    [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],#TXM
    [3,4,3,2,5,3,4,2,2,2,4,5,3,5,3,3,5,5,2,3], #IXR
    [3,4,3,4,4,3,4,3,3,3,4,4,4,4,3,3,4,4,4,3], #XXK
    [4,4,4,4,5,5,5,1,2,2,5,5,5,5,2,5,2,5,1,4], #PXG

    [3,4,3,4,4,2,4,2,2,2,4,4,4,4,2,3,4,4,2,3], #KXP

    [2,5,3,2,4,3,4,2,3,3,4,4,5,4,2,2,4,5,2,3] #SXT
]



# Initialize lists to store positive and negative subscale scores for both groups
group1_positive_scores = []
group1_negative_scores = []

group2_positive_scores = []
group2_negative_scores = []

# Loop through each item and its corresponding response for Group 1
for participant_responses in group1_responses:
    # Initialize lists to store positive and negative subscale scores for the participant
    participant_positive_scores = []
    participant_negative_scores = []

    # Loop through each item and its corresponding response
    for i, response in enumerate(participant_responses):
        item_category = item_categorization[i]

        # Check if the item is categorized as "Pos" or "Neg" and assign a score accordingly
        if item_category == "Pos":
            participant_positive_scores.append(response)
        elif item_category == "Neg":
            # Reverse the score for negative items
            participant_negative_scores.append(6 - response)  # Reverse the score from 1 to 5 to 5 to 1

    # Calculate the mean (average) scores for the positive and negative subscales for the participant in Group 1
    mean_positive = sum(participant_positive_scores) / len(participant_positive_scores)
    mean_negative = sum(participant_negative_scores) / len(participant_negative_scores)

    # Append the participant's mean scores to the group's lists
    group1_positive_scores.append(mean_positive)
    group1_negative_scores.append(mean_negative)


import numpy as np  # Import numpy for mean and standard deviation calculation

group1_diff_scores = []  # List to store the difference scores for each participant

for participant_responses in group1_responses:
    participant_positive_scores = []
    participant_negative_scores = []

    # Loop through each response and categorize
    for i, response in enumerate(participant_responses):
        item_category = item_categorization[i]
        
        if item_category == "Pos":
            participant_positive_scores.append(response)
        elif item_category == "Neg":
            participant_negative_scores.append(6 - response)  # Reverse scoring for negative items

    # Calculate mean scores for positive and negative subscales for the participant
    mean_positive = sum(participant_positive_scores) / len(participant_positive_scores)
    mean_negative = sum(participant_negative_scores) / len(participant_negative_scores)

    # Calculate the difference and store it
    diff_score = mean_positive - mean_negative
    group1_diff_scores.append(diff_score)

# Calculate the mean and standard deviation of the differences for the group
mean_diff = np.mean(group1_diff_scores)
std_diff = np.std(group1_diff_scores)



print("NO TL MEAN LAST",mean_diff)
print("NO TL STD LAST",std_diff)


# Loop through each item and its corresponding response for Group 2
for participant_responses in group2_responses:
    # Initialize lists to store positive and negative subscale scores for the participant
    participant_positive_scores = []
    participant_negative_scores = []

    # Loop through each item and its corresponding response
    for i, response in enumerate(participant_responses):
        item_category = item_categorization[i]

        # Check if the item is categorized as "Pos" or "Neg" and assign a score accordingly
        if item_category == "Pos":
            participant_positive_scores.append(response)
        elif item_category == "Neg":
            # Reverse the score for negative items
            participant_negative_scores.append(6 - response)  # Reverse the score from 1 to 5 to 5 to 1

    # Calculate the mean (average) scores for the positive and negative subscales for the participant in Group 2
    mean_positive = sum(participant_positive_scores) / len(participant_positive_scores)
    mean_negative = sum(participant_negative_scores) / len(participant_negative_scores)

    # Append the participant's mean scores to the group's lists
    group2_positive_scores.append(mean_positive)
    group2_negative_scores.append(mean_negative)

# Calculate the total means for positive and negative subscales for both groups
total_mean_positive_group1 = sum(group1_positive_scores) / len(group1_positive_scores)
total_mean_negative_group1 = sum(group1_negative_scores) / len(group1_negative_scores)

total_mean_positive_group2 = sum(group2_positive_scores) / len(group2_positive_scores)
total_mean_negative_group2 = sum(group2_negative_scores) / len(group2_negative_scores)

# Calculate the difference between total mean positive and total mean negative scores for both groups
difference_group1 = total_mean_positive_group1 - total_mean_negative_group1
difference_group2 = total_mean_positive_group2 - total_mean_negative_group2




group2_diff_scores = []  # List to store the difference scores for each participant

for participant_responses in group2_responses:
    participant_positive_scores = []
    participant_negative_scores = []

    # Loop through each response and categorize
    for i, response in enumerate(participant_responses):
        item_category = item_categorization[i]
        
        if item_category == "Pos":
            participant_positive_scores.append(response)
        elif item_category == "Neg":
            participant_negative_scores.append(6 - response)  # Reverse scoring for negative items

    # Calculate mean scores for positive and negative subscales for the participant
    mean_positive = sum(participant_positive_scores) / len(participant_positive_scores)
    mean_negative = sum(participant_negative_scores) / len(participant_negative_scores)

    # Calculate the difference and store it
    diff_score = mean_positive - mean_negative
    group2_diff_scores.append(diff_score)

# Calculate the mean and standard deviation of the differences for the group
mean_diff = np.mean(group2_diff_scores)
std_diff = np.std(group2_diff_scores)



print("TL MEAN LAST",mean_diff)
print("TL STD LAST",std_diff)



# Print the results for both groups
print("Group 1 - Total Mean Positive Subscale Score:", total_mean_positive_group1)
print("Group 1 - Total Mean Negative Subscale Score:", total_mean_negative_group1)
print("Group 1 - Difference between Total Mean Positive and Negative Subscale Scores:", difference_group1)

print("Group 2 - Total Mean Positive Subscale Score:", total_mean_positive_group2)
print("Group 2 - Total Mean Negative Subscale Score:", total_mean_negative_group2)
print("Group 2 - Difference between Total Mean Positive and Negative Subscale Scores:", difference_group2)
