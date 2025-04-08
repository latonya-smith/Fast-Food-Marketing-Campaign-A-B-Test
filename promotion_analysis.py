import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


import statsmodels.api as sm
import statsmodels.formula.api as smf


processed_data_path = "C:/Users/lsmith3/OneDrive - MBTA/Documents/Personal Projects/AB Testing/cleaned_WA_Marketing-Campaign.csv"
df = pd.read_csv(processed_data_path)

print("Showing basic Descriptive Statistics...\n")
print("Dataset contains {} rows and {} colums". format(df.shape[0], df.shape[1]))
print("Sample data:\n", df.head())
print("Data Summary:\n", df.describe())

print("Changing columns : Promotion, HighPerformer, LocationID, MarketID and week into categorical varibales \n")
df['Promotion'] = df['Promotion'].astype('category')
df['HighPerformer'] = df['HighPerformer'].astype('category')
df['LocationID'] = df['LocationID'].astype('category')
df['MarketID'] = df['MarketID'].astype('category')
df['week'] = df['week'].astype('category')

print(df.info(), "\n")

print("Descriptive Statistics: Grouping by Promotion... \n")
sales_summary = df.groupby('Promotion')['SalesInThousands'].agg(['mean', 'std', 'count'])

print(sales_summary)

print("One-Way Anova to test if there is a significant difference in sales across promotions")
print("ANOVA results ...\n")

anova_result = stats.f_oneway(
    df[df['Promotion'] == 1]['SalesInThousands'],
    df[df['Promotion'] == 2]['SalesInThousands'],
    df[df['Promotion'] == 3]['SalesInThousands']
)

print("F-statistic: {},  p-value: {} \n".format(anova_result.statistic, anova_result.pvalue))

print("Interpretation\n")

if anova_result.pvalue < 0.05:
    print("There is a statistically significant difference in sales across promotions. ")
else:
    print("There is no statistically significant difference in sales across promotions. ")
    
    
print("Visualizing sales by promotion... ")
plt.figure(figsize=(10,6))
sns.boxplot(df, x='Promotion', y='SalesInThousands', hue='Promotion', palette='Set2', legend=False)
plt.title('Sales Distribution by Promotion')
plt.xlabel('Promotion')
plt.ylabel('SalesInThousands')
plt.show()


print("Calculating uplift for each promotion (compared to Control Group) \n")
print("Promotion 1 is being used as the Control Group \n")

control_mean = df[df['Promotion']== 2]['SalesInThousands'].mean()
#Percentage Uplift
df['Uplift'] = (df['SalesInThousands'] - control_mean) / control_mean

plt.figure(figsize=(10,6))
sns.barplot(df, x='Promotion', y='Uplift', palette='Set1')
plt.title('Uplift in Sales by Promotion(Compared to Control: Promotion 2)')
plt.xlabel('Promotion')
plt.ylabel('Uplift')
plt.show()

print("""
Promotion 2 (Baseline): Since Promotion 2 is the baseline, its uplift is considered 0%.

Promotion 1: The uplift for Promotion 1 is positive, ranging from 0% to 23%, with a vertical line indicating the 95% confidence interval (from 17% to 29%). This suggests that Promotion 1 outperformed Promotion 2, with a notable positive impact on sales, indicating it was significantly more effective than Promotion 2.

Promotion 3: Promotion 3 also shows a positive uplift ranging from 0% to 17%, with a vertical line indicating the 95% confidence interval (from 12% to 23%). This suggests that Promotion 3 also performed better than Promotion 2, though not as strongly as Promotion 1.
""")


print("Additional Analyses... \n")

print("Sales over the 4 week period for top 5 locations.\n")

# Select top 5 LocationIDs based on total sales
top_locations = df.groupby('LocationID')['SalesInThousands'].sum().nlargest(5).index

# Filter the DataFrame to include only the rows with top 5 LocationIDs
df_filtered = df[df['LocationID'].isin(top_locations)]

# Define a custom color palette with exactly 5 colors
colors = sns.color_palette("Set2", n_colors=5)

# Create a plot for each LocationID with a unique color
plt.figure(figsize=(24, 10))

# Loop through each of the top 5 LocationIDs and plot them separately
for idx, location in enumerate(top_locations):
    location_data = df_filtered[df_filtered['LocationID'] == location]
    sns.lineplot(data=location_data, x='week', y='SalesInThousands', 
                 label=location, color=colors[idx], marker='o')

# Set the plot's title and labels
plt.title('Impact of Promotion Over 4 Weeks (Top 5 LocationIDs)', fontsize=18)
plt.xlabel('Week', fontsize=14)
plt.ylabel('Sales (in Thousands)', fontsize=14)

# Show the legend with LocationID labels
plt.legend(title='Top 5 LocationIDs', bbox_to_anchor=(1.05, 1), loc='upper center')

# Show the plot
plt.show()


print("Older vs Younger Store Promotion Performance Over 4 Weeks. \n")
# Categorize the AgeOfStore into Older and Younger stores
median_age = df['AgeOfStore'].median()
df['StoreCategory'] = df['AgeOfStore'].apply(lambda x: 'Older' if x >= median_age else 'Younger')

# Now you can plot the impact of promotion for both categories
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='week', y='SalesInThousands', hue='StoreCategory', style='Promotion', markers=True, ci=None)
plt.title('Impact of Promotion on Older vs Younger Stores Over 4 Weeks')
plt.xlabel('Week')
plt.ylabel('Sales (in Thousands)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper center')
plt.show()


#Promotion Effect by Market Size:
print("Promotion Effect by Market Size..\n")
# Grouping by MarketSize and Promotion to see average sales
promo_by_market_size = df.groupby(['MarketSize', 'Promotion'])['SalesInThousands'].mean().reset_index()

# Visualize the effectiveness of promotions in different market sizes
plt.figure(figsize=(10, 6))
sns.barplot(data=promo_by_market_size, x='MarketSize', y='SalesInThousands', hue='Promotion', palette='Set2')
plt.title('Promotion Effectiveness by Market Size')
plt.xlabel('Market Size')
plt.ylabel('Average Sales (in Thousands)')
plt.show()


# Predicting Sales with Linear Regression
print(" Predicting Sales with GLM... \n")
formula = 'SalesInThousands ~ C(Promotion) + C(MarketSize) + AgeOfStore + marketSizeEncoded'

#Gaussian for continuous data
model = smf.glm(formula=formula, data=df, family=sm.families.Gaussian()).fit()

print(model.summary())

print("""
Predicting Sales with GLM...

The Generalized Linear Model (GLM) results show the following insights:

- The intercept value indicates that, for the baseline categories, sales start at approximately 18.65 thousand units.
- **Promotion 2** (compared to Promotion 1) has a significant negative effect on sales, with a decrease of around 10.75 thousand units.
- **Promotion 3** does not have a statistically significant impact on sales, with a p-value of 0.351.
- **MarketSize**: Being in a medium-sized market significantly decreases sales by about 8.18 thousand units, while being in a small market significantly increases sales by approximately 22.84 thousand units.
- **Age of Store** has no significant impact on sales (p-value = 0.320), suggesting store age doesn't affect sales in this model.
- **marketSizeEncoded**: A one-unit increase in this encoded market size feature is associated with a significant sales increase of 18.45 thousand units.

Overall, the model explains 74.77% of the variance in sales based on these variables.
""")

