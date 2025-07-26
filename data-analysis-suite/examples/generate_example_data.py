import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# Generate example dataset
n_rows = 200

# Create random dates
dates = [datetime.now() - timedelta(days=random.randint(1, 1000)) for _ in range(n_rows)]

# Create random data
data = {
    'date': dates,
    'temperature': np.random.normal(loc=20, scale=5, size=n_rows),
    'humidity': np.random.normal(loc=60, scale=10, size=n_rows),
    'weather_type': random.choices(['Sunny', 'Cloudy', 'Rainy', 'Snowy'], k=n_rows),
    'city': random.choices(['New York', 'Los Angeles', 'Chicago', 'Texas'], k=n_rows)
}

# Create DataFrame
example_df = pd.DataFrame(data)

# Save to CSV
example_df.to_csv('example_data.csv', index=False)

print("Example data generated and saved to 'example_data.csv'")
