age = int(input("Enter your age: "))
citizenship = input("Enter your citizenship: ").lower()
if age >= 18 and citizenship == "yes":
    print("You have the right to vote")
elif age < 18 and citizenship == "yes":
    print(f"You are not eligible to vote. you need to wait {18-age} year ")
elif age >= 18 and citizenship == "no":
    print("You do not have the right to vote")
else:
    print("Not eligible to vote")