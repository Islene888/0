
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
file_path = 'D:\\Notets\\python\\Data Science\\HW7\\amr_ds.csv'
data = pd.read_csv(file_path)

# Naive Bayes Classification for Predicting 'Not_MDR'
X = data[['Ampicillin', 'Penicillin']]
y = data['Not_MDR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Naive Bayes Model Accuracy:', accuracy)

# Calculating Specific Values (amp_pen, amp_nmdr, pen_nmdr)
amp_pen = ((data['Ampicillin'] == 1) & (data['Penicillin'] == 1)).sum()
amp_nmdr = ((data['Ampicillin'] == 1) & (data['Not_MDR'] == 1)).sum()
pen_nmdr = ((data['Penicillin'] == 1) & (data['Not_MDR'] == 1)).sum()
print('amp_pen:', amp_pen, 'amp_nmdr:', amp_nmdr, 'pen_nmdr:', pen_nmdr)

# Simplified Approach for Predicting State Sequence
state_mapping = ['Ampicillin', 'Penicillin', 'Not_MDR']
prob_sequences = {
    'Ampicillin': [0.6, 0.4, 0.6],
    'Penicillin': [0.7, 0.3, 0.7],
    'Not_MDR': [0.2, 0.8, 0.2]
}
initial_prob = 1 / 3
state_probs_corrected = {}
for state in state_mapping:
    state_prob_sequence = np.prod(prob_sequences[state]) * initial_prob
    state_probs_corrected[state] = state_prob_sequence
most_probable_state_each_observation = max(state_probs_corrected, key=state_probs_corrected.get)
print('Most Probable State:', most_probable_state_each_observation)

# Predicting the most probable sequence of states for a given sequence of observations
observations = ['Infection after surgery', 'No infection after surgery', 'Infection after surgery']

# Mapping of observations to their corresponding states
# This needs to be defined based on the problem context. Here's a hypothetical example:
observation_to_state_probabilities = {
    'Infection after surgery': {'Ampicillin': 0.6, 'Penicillin': 0.7, 'Not_MDR': 0.2},
    'No infection after surgery': {'Ampicillin': 0.4, 'Penicillin': 0.3, 'Not_MDR': 0.8}
}

# Function to predict the most probable state for each observation
def predict_states(sequence, mapping):
    predicted_states = []
    for observation in sequence:
        if observation in mapping:  # Check if the observation is one we can map
            # Find the state with the highest probability for the current observation
            max_prob_state = max(mapping[observation], key=mapping[observation].get)
            predicted_states.append(max_prob_state)
        else:
            predicted_states.append('Unknown')  # If the observation is not recognized
    return predicted_states

# Predicting states for the given sequence of observations
predicted_sequence = predict_states(observations, observation_to_state_probabilities)
print('Predicted Sequence of States:', predicted_sequence)

