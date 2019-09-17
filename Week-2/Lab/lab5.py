# There are three seating categories at a stadium. For a football games,
# Class A seat’s cost €25, Class B seat’s cost €20 and Class C seat’s cost €25.
# Write a program that asks how many tickets for each class of seats were sold, and then display
# the amount of income generated from ticket sales.

a_tickets = int(input("How many Class A tickets were sold?: "))
b_tickets = int(input("How many Class B tickets were sold?: "))
c_tickets = int(input("How many Class C tickets were sold?: "))

income = (a_tickets + c_tickets) * 25 + b_tickets * 20

print(f"Total income from tickets sale is $ {income}")
