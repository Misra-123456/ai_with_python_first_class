balance = 1000
while True:
    print("enter choice: balance / deposit / withdraw / exit")
    choice = input("Enter your choice: ").lower()
    if choice == "balance":
        print(f"Your balance is: {balance}")
    elif choice == "deposit":
        deposit = float(input("amount to deposit: "))
        balance += deposit
        print(f"{deposit} deposited. New balance:{balance}")
    elif choice == "withdraw":
        withdraw = float(input("Enter amount to withdraw: "))
        if withdraw > balance:
            print("Insufficient balance")
        else:
            balance -= withdraw
            print(f"{withdraw} withdrawn. New balance: {balance}")
    elif choice == "exit":
        print("Thank you for using the ATM. Goodbye!")
        break
    else:
        print("Invalid option. Please type: balance / deposit / withdraw / exit")
