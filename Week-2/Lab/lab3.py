# Write a program that will ask a student for their first name and then for their surname.
# It should then ask the student to enter the int numerical grade they received in each of their three subjects.
# The program should then print out the full name of the student along with their average numerical grade
# (Use only a single print statement)

first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")
grade1 = int(input("Enter your grade for subject 1: "))
grade2 = int(input("Enter your grade for subject 2: "))
grade3 = int(input("Enter your grade for subject 3: "))

print(f"{first_name} {last_name}'s average grade is {(grade1+grade2+grade3)/3}")
