import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
# Replace 'covid_data.csv' with the actual file path
covid_data = pd.read_csv('covid-19-cases.csv')

# Display some rows of the dataframe
print("First few rows of the dataframe:")
print(covid_data.head())

# Visualization
# Example 1: Line plot of confirmed cases over time for a specific country
def plot_confirmed_cases_over_time(country_name, skip_dates=5):
    country_data = covid_data[covid_data['Country/Region'] == country_name]
    dates = country_data.columns[3:]  # assuming dates start from the 4th column
    confirmed_cases = country_data.iloc[:, 3:].sum(axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(dates[::skip_dates], confirmed_cases[::skip_dates], marker='o', linestyle='-')
    plt.title(f'Confirmed Cases Over Time in {country_name}')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage:
plot_confirmed_cases_over_time('US', skip_dates=10)

