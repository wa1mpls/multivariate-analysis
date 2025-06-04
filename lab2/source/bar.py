import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
covid_data = pd.read_csv('covid-19-cases.csv')

# Display some rows of the dataframe
print("First few rows of the dataframe:")
print(covid_data.head())

# Visualization
# Bar plot showing the increase of COVID-19 cases in the US over time for a few rows of data
def plot_us_cases_over_time():
    us_data = covid_data[covid_data['Country/Region'] == 'US'].iloc[:5]  # Selecting first 5 rows for demonstration
    dates = us_data.columns[3::10]  # Skip 10 dates
    us_cases = us_data.iloc[:, 3::10].sum(axis=0)  # Skip 10 dates

    plt.figure(figsize=(10, 6))
    plt.bar(dates, us_cases, color='skyblue')
    plt.title('Increase of COVID-19 Cases in the US Over Time (Skipping 10 dates)')
    plt.xlabel('Date')
    plt.ylabel('Total Cases')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('us_cases_over_time_bar.png')  # Save the bar plot as a PNG image
    plt.close()  # Close the figure after saving


# Example usage:
plot_us_cases_over_time()

