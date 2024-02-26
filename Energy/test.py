import pandas as pd  # Import Pandas library for data processing
import numpy as np  # Import NumPy library for numerical calculations
from scipy.optimize import linprog  # Import linear programming function from SciPy library
import matplotlib.pyplot as plt  # Import Matplotlib library for result visualization

# Assumed data
TECHNOLOGY_COSTS = {'solar': 20, 'wind': 25, 'gas': 30, 'storage': 40}  # Costs of different technologies
TRANSMISSION_LIMITS = {'region1_region2': 100, 'region2_region3': 80}  # Transmission limits between different regions
COMMODITY_PRICES = {'natural_gas': 50, 'coal': 60}  # Commodity prices
POLICY_FACTORS = {'carbon_tax': 30, 'renewable_incentive': 10}  # Policy factors
LOAD_GROWTH = {'base_load': 100, 'electrification_impact': 20}  # Load growth rates
RENEWABLE_TARGETS = {'solar': 50, 'wind': 30}  # Renewable energy targets
SELF_CONSUMPTION_RATES = {'solar': 0.3, 'wind': 0.1}  # Self-consumption rates

# Data input and preprocessing
class DataInputHandler:
    def __init__(self, filepath):
        self.filepath = filepath  # File path
        self.raw_data = None  # Raw data

    def load_data(self):
        # Load data using Pandas
        self.raw_data = pd.read_csv(self.filepath)

    def preprocess_data(self):
        # Preprocess data, use forward fill to handle missing values and convert to NumPy array
        self.processed_data = self.raw_data.ffill().values

# Extended capacity expansion model
class ExtendedCapacityExpansionModel:
    def __init__(self, data):
        self.data = data  # Input data
        self.results = None  # Optimization results
        self.cost_vector = None  # Cost vector
        self.A_eq = None  # Equality constraint coefficient matrix
        self.b_eq = None  # Equality constraint values
        self.bounds = None  # Variable bounds

    def calculate_cost_vector(self):
        # Calculate cost vector considering costs of different technologies, commodity prices, and policy impacts
        self.cost_vector = np.array([
            TECHNOLOGY_COSTS['solar'] + POLICY_FACTORS['carbon_tax'] * SELF_CONSUMPTION_RATES['solar'],
            TECHNOLOGY_COSTS['wind'] + POLICY_FACTORS['carbon_tax'] * SELF_CONSUMPTION_RATES['wind'],
            TECHNOLOGY_COSTS['gas'] + COMMODITY_PRICES['natural_gas'],
            TECHNOLOGY_COSTS['storage'],
            COMMODITY_PRICES['coal'] + POLICY_FACTORS['carbon_tax']
        ])

    def set_constraints(self):
        # Set constraints including total capacity of renewable and non-renewable energy sources
        self.A_eq = np.array([
            [1, 1, 0, 0, 0],  # Total renewable energy capacity
            [0, 0, 1, 1, 1]   # Total non-renewable energy capacity
        ])
        self.b_eq = np.array([
            RENEWABLE_TARGETS['solar'],  # Renewable energy demand
            LOAD_GROWTH['base_load'] + LOAD_GROWTH['electrification_impact'] - RENEWABLE_TARGETS['solar']  # Non-renewable demand
        ])

    def set_bounds(self):
        # Set capacity bounds for different technologies
        self.bounds = [
            (20, RENEWABLE_TARGETS['solar'] + 30),  # Solar
            (15, RENEWABLE_TARGETS['wind'] + 40),  # Wind
            (10, 200),                             # Gas
            (5, TRANSMISSION_LIMITS['region1_region2'] + 20),  # Storage
            (10, TRANSMISSION_LIMITS['region2_region3'] + 30)  # Coal
        ]

    def optimize_capacity(self):
        # Perform linear programming optimization using SciPy
        self.results = linprog(c=self.cost_vector, A_eq=self.A_eq, b_eq=self.b_eq, bounds=self.bounds, method='highs')

    def post_optimization_analysis(self):
        # Analyze optimization results, if successful, save optimized capacity allocation
        if self.results.success:
            self.optimized_capacity = self.results.x
        else:
            self.optimized_capacity = None
            print("Optimization failed.")

# Result visualization
class ResultVisualizer:
    @staticmethod
    def visualize_results(capacity):
        if capacity is not None:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(capacity)), capacity, color='skyblue')
            plt.xlabel('Technologies')
            plt.ylabel('Optimized Capacity')
            plt.title('Capacity Expansion Results')
            plt.xticks(range(len(capacity)), ['Solar', 'Wind', 'Gas', 'Storage', 'Coal'])
            plt.show()
        else:
            print("No results to visualize.")

    @staticmethod
    def visualize_market_trends(trends):
        plt.figure(figsize=(10, 6))
        for trend in trends:
            plt.plot(trend['years'], trend['values'], label=trend['name'])
        plt.xlabel('Year')
        plt.ylabel('Market Value')
        plt.title('Market Trends')
        plt.legend()
        plt.show()

    @staticmethod
    def visualize_roadmap(roadmap):
        plt.figure(figsize=(10, 6))
        # Ensure the roadmap is a dictionary with years as keys and events as values
        for year, event in roadmap.items():
            plt.plot(year, 1, 'ro')  # Plot a red dot at each year
            plt.text(year, 1.05, event, ha='center', va='bottom')  # Display the event text above the dot
        plt.ylim(0.8, 1.2)  # Set y-axis range to avoid overlap between text and dots
        plt.xlabel('Time')
        plt.title('Development Roadmap')
        plt.yticks([])  # Hide y-axis labels
        plt.show()

# Main program
def main():
    # Initialize data handler and load/preprocess data
    data_handler = DataInputHandler('input_data.csv')
    data_handler.load_data()
    data_handler.preprocess_data()

    # Initialize and run the extended capacity expansion model
    cam = ExtendedCapacityExpansionModel(data_handler.processed_data)
    cam.calculate_cost_vector()
    cam.set_constraints()
    cam.set_bounds()
    cam.optimize_capacity()
    cam.post_optimization_analysis()

    # Visualization of results
    ResultVisualizer.visualize_results(cam.optimized_capacity)

    # Example data for additional visualizations
    market_trends = [
        {'years': [2020, 2021, 2022, 2023], 'values': [100, 110, 105, 115], 'name': 'Solar'},
        {'years': [2020, 2021, 2022, 2023], 'values': [80, 90, 85, 95], 'name': 'Wind'}
    ]
    ResultVisualizer.visualize_market_trends(market_trends)

    roadmap = {
        2020: 'Project Start',
        2021: 'Phase 1 Completion',
        2022: 'Phase 2 Start'
    }
    ResultVisualizer.visualize_roadmap(roadmap)

if __name__ == '__main__':
    main()
