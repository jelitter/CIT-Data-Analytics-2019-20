# Write a simple program that will ask the user to enter a number that is a multiple of 10.
# If the number entered is not a multiple of 10 then the program should display an error message

num = input("Enter a number multiple of 10: ")

if int(num) % 10 != 0:
    print("Error: the number is not multiple of 10.")
