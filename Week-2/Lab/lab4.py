# Write a program to calculate and display a person’s body mass index(BMI).
# A persons BMI is calculated with the following formula:
#   • BMI = (weight/height2) * 703
# Where weight in in pounds and height in in inches.


weight = float(input("Enter your weight in pounds: "))
height = float(input("Enter your height in inches: "))
bmi = (weight/height*2)*703

print(f"Your BMI is {bmi}")
