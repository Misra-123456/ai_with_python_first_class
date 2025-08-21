age = int(input("Enter your age: "))
if 15 <= age < 18:
    weight = float(input("Enter your weight: "))
    if weight >= 55:
        print(".")
    else:
        print("Not eligible")
elif age >= 18:
    print("You are eligible (no weight check needed).")
else:
    print("Not eligible: age is below 15.")
