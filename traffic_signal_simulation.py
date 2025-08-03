import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pygame
import random
import time
# Load datasets
data = pd.read_csv("traffic_signal_data.csv")
data_extra = pd.read_csv("traffic_signal_data_extra.csv")

# Combine datasets
data = pd.concat([data, data_extra], ignore_index=True)

# Encode light_state to numeric labels for ML model
data['light_state'] = data['light_state'].map({'GREEN': 0, 'YELLOW': 1, 'RED': 2})

# Split features and target
X = data[['vehicle_count', 'pedestrian_count', 'ambulance_present']]
y = data['light_state']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
# Define Q-learning parameters
q_table = np.zeros((51, 21, 2))  # (vehicle_count, pedestrian_count, action)
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 1)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

def update_q_table(state, action, reward, new_state):
    q_table[state][action] = q_table[state][action] + alpha * (
        reward + gamma * np.max(q_table[new_state]) - q_table[state][action]
)

# Simulate an RL episode
def rl_episode(vehicle_count, pedestrian_count):
    state = (vehicle_count, pedestrian_count)
    action = choose_action(state)
    # Reward logic: prioritize green light for ambulances or low pedestrian density
    reward = 1 if (action == 0 and pedestrian_count < 5) else -1
    update_q_table(state, action, reward, state)
    return action  # 0 = Green, 1 = Yellow
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Traffic Signal Simulation")

# Colors for signals and background
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Load and train the ML model using RandomForest
data = pd.read_csv("traffic_signal_data.csv")
data['light_state'] = data['light_state'].map({'GREEN': 0, 'YELLOW': 1, 'RED': 2})

X = data[['vehicle_count', 'pedestrian_count', 'ambulance_present']]
y = data['light_state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Traffic Signal Simulation")
font = pygame.font.Font(None, 36)  # Font for rendering text

# Function to draw the traffic signal
def draw_signal(color):
    pygame.draw.circle(screen, color, (400, 150), 100)
    pygame.display.flip()  # Update the display immediately

# Function to display logs and timer in the simulation
"""def display_log_and_timer(log, timer=None):
    screen.fill(BLACK)  # Clear the screen

    # Draw the signal circle as red if in countdown mode
    if timer is not None:
        draw_signal(RED)  # Keep the light red for the entire countdown
    else:
        draw_signal(WHITE)  # Default state when not in countdown

    # Render and display the latest log entry
    log_surface = font.render(log, True, WHITE)
    screen.blit(log_surface, (10, 300))  # Display the log at a fixed position

    # Display the countdown timer during the red light phase
    if timer is not None:
        timer_surface = font.render(f"Red Light Timer: {timer}", True, RED)
        screen.blit(timer_surface, (10, 350))  # Display timer just below the log

    pygame.display.flip()  # Update the display"""

# Function to display logs and timer in a table format in the simulation
def display_log_and_timer(log_count, vehicle_count, pedestrian_count, ambulance_present, signal_state, timer=None):
    screen.fill(BLACK)  # Clear the screen

    # Draw the signal circle as red if in countdown mode
    if timer is not None:
        draw_signal(RED)  # Keep the light red for the entire countdown
    else:
        draw_signal(WHITE)  # Default state when not in countdown

    # Display the header
    header = font.render(f"Log #{log_count}:", True, WHITE)
    screen.blit(header, (10, 250))  # Position header near the top of the log section

    # Display column names
    column_names = font.render("Vehicles    Pedestrians    Ambulance         Signal", True, WHITE)
    screen.blit(column_names, (10, 300))

    # Display row values
    row_values = font.render(
        f"    {vehicle_count}                   {pedestrian_count}                      {ambulance_present}            {signal_state}", True, WHITE
    )
    screen.blit(row_values, (10, 340))

    # Display the countdown timer during the red light phase
    if timer is not None:
        timer_surface = font.render(f"Red Light Timer: {timer}", True, RED)
        screen.blit(timer_surface, (10, 380))  # Display timer below the table

    pygame.display.flip()  # Update the display


"""# Main simulation loop with ML integration and log display
def run_simulation():
    running = True
    log_count = 1  # Log serial number

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Generate random real-time data (replace with sensor input if needed)
        vehicle_count = random.randint(0, 50)
        pedestrian_count = random.randint(0, 20)
        ambulance_present = random.randint(0, 1)

        # Predict the signal state using the ML model
        # Create DataFrame with feature names for consistent prediction
        input_data = pd.DataFrame([[vehicle_count, pedestrian_count, ambulance_present]], 
                          columns=['vehicle_count', 'pedestrian_count', 'ambulance_present'])
        prediction = model.predict(input_data)[0]

        signal_state = ['GREEN', 'YELLOW', 'RED'][prediction]

        # Calculate red light timer based on pedestrian count
        red_light_timer = 6 if pedestrian_count >= 10 else 3

        # Create the log entry
        log_entry = (f"Log #{log_count}: Vehicles: {vehicle_count}, "
                     f"Pedestrians: {pedestrian_count}, Ambulance: {ambulance_present}, "
                     f"Signal: {signal_state}, Red Timer: {red_light_timer}s")
        log_count += 1  # Increment the log count

        # Display the log entry in the simulation window
        display_log_and_timer(log_entry)

        # If the prediction is GREEN and there are no pedestrians, skip to the next log
        if prediction == 0 and pedestrian_count == 0:
            draw_signal(GREEN)
            time.sleep(2)  # Display the green signal briefly
            continue  # Move directly to the next iteration

        # Simulate the traffic light behavior with correct transitions
        if prediction == 0:  # GREEN light
            draw_signal(GREEN)
            time.sleep(3)
            draw_signal(YELLOW)
            time.sleep(3)
            draw_signal(RED)  # Now go to RED before returning to GREEN
            # Display red light timer countdown
            for timer in range(red_light_timer, 0, -1):
                display_log_and_timer(log_entry, timer)
                time.sleep(1)  # Countdown timer
            pedestrian_count = 0  # Reset pedestrian count after red

        elif prediction == 1:  # YELLOW light
            draw_signal(YELLOW)
            time.sleep(3)
            draw_signal(RED)  # From YELLOW, it must go to RED
            # Display red light timer countdown
            for timer in range(red_light_timer, 0, -1):
                display_log_and_timer(log_entry, timer)
                time.sleep(1)  # Countdown timer
            pedestrian_count = 0  # Reset pedestrian count after red

        elif prediction == 2:  # RED light
            draw_signal(RED)
            # Display red light timer countdown
            for timer in range(red_light_timer, 0, -1):
                display_log_and_timer(log_entry, timer)
                time.sleep(1)  # Countdown timer
            pedestrian_count = 0  # Reset pedestrian count after red

        # After RED, always return to GREEN
        draw_signal(GREEN)
        time.sleep(3)

    pygame.quit()"""


# Main simulation loop with ML integration and table log display
def run_simulation():
    running = True
    log_count = 1  # Log serial number

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Generate random real-time data (replace with sensor input if needed)
        vehicle_count = random.randint(0, 50)
        pedestrian_count = random.randint(0, 20)
        ambulance_present = random.randint(0, 1)

        # Predict the signal state using the ML model
        input_data = pd.DataFrame([[vehicle_count, pedestrian_count, ambulance_present]], 
                          columns=['vehicle_count', 'pedestrian_count', 'ambulance_present'])
        prediction = model.predict(input_data)[0]

        signal_state = ['GREEN', 'YELLOW', 'RED'][prediction]

        # Calculate red light timer based on pedestrian count
        red_light_timer = 6 if pedestrian_count >= 10 else 3

        # Display the log entry in the table format in the simulation window
        display_log_and_timer(log_count, vehicle_count, pedestrian_count, ambulance_present, signal_state, None)
        log_count += 1  # Increment the log count

        # Simulate the traffic light behavior with correct transitions
        if prediction == 0:  # GREEN light
            draw_signal(GREEN)
            time.sleep(3)
            draw_signal(YELLOW)
            time.sleep(3)
            draw_signal(RED)  # Now go to RED before returning to GREEN
            for timer in range(red_light_timer, 0, -1):
                display_log_and_timer(log_count, vehicle_count, pedestrian_count, ambulance_present, signal_state, timer)
                time.sleep(1)
            pedestrian_count = 0

        elif prediction == 1:  # YELLOW light
            draw_signal(YELLOW)
            time.sleep(3)
            draw_signal(RED)
            for timer in range(red_light_timer, 0, -1):
                display_log_and_timer(log_count, vehicle_count, pedestrian_count, ambulance_present, signal_state, timer)
                time.sleep(1)
            pedestrian_count = 0

        elif prediction == 2:  # RED light
            draw_signal(RED)
            for timer in range(red_light_timer, 0, -1):
                display_log_and_timer(log_count, vehicle_count, pedestrian_count, ambulance_present, signal_state, timer)
                time.sleep(1)
            pedestrian_count = 0

        draw_signal(GREEN)
        time.sleep(3)

    pygame.quit()

# Run the simulation
run_simulation()
