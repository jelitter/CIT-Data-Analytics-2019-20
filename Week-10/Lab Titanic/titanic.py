import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("titanic.csv", sep='\t', header=(0))

# filter the dataframe to only return rows
# where the passenger died
# criteria_survived = df['Survived'] == 0
# criteria_died = df['Survived'] == 1

# print(df.head())

# survivors = df[criteria_survived]
# fatalities = df[criteria_died]

# passenger_class_group = fatalities.groupby("Pclass")

# # extract the Survived column from the groupby object
# classSurvived = passenger_class_group['Survived'].count()
# print(classSurvived)
# classSurvived.plot()
# plt.show()


# a.Create a bar graph to show the number of males and females who have died and survived.

bar_width = 0.45

# criteria_survived = df['Survived'] == 0
# criteria_died = df['Survived'] == 1

criteria_male = df['Sex'] == 'male'
criteria_female = df['Sex'] == 'female'

# survivors = df[criteria_survived]
# fatalities = df[criteria_died]

males = df[criteria_male]
females = df[criteria_female]

# passenger_fatalities_sex_group = fatalities.groupby("Sex")
# passenger_survivors_sex_group = survivors.groupby("Sex")

passenger_male_survival = males.groupby("Survived")
passenger_female_survival = females.groupby("Survived")

# class_fatalities_by_sex = passenger_fatalities_sex_group['Survived'].count()
# class_survivors_by_sex = passenger_survivors_sex_group['Survived'].count()

class_survival_male = passenger_male_survival['Sex'].count()
class_survival_female = passenger_female_survival['Sex'].count()

# print(class_fatalities_by_sex)
# print(class_survivors_by_sex)

print(class_survival_male)
print(class_survival_female)

# plt.bar(['male', 'female'], class_fatalities_by_sex, bar_width)
# plt.bar(['male', 'female'], class_survivors_by_sex, bar_width)

plt.bar(['died', 'survived'], class_survival_male, bar_width)
plt.bar(['died', 'survived'], class_survival_female, bar_width)

plt.show()


# b.Create a bar graph and show the number of males and females in the age range[20, 40]
# c.Use a plot and show if there is any relationship between the number of survived passengers and their class ticket.
# d.Use a plot and show the relation between fare and class ticket.
# e.Use a pie chart and visualize the population among three different class ticket.
