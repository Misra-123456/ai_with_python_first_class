def tester(giventext="too short"):
    print(giventext)

def main():
    while True:
        text=input("write something(quite ends): ")
        if text == "quit":
            break
        if len(text) < 10:
            tester()
        else:
            print(text)
main()



shopping_list=[]
while True:
    option = input("do you want to\n1.Add or\n2.Remove or\n3.Quit?: ")
    if option == "1":
        item = input("enter item: ")
        shopping_list.append(item)
        print(shopping_list)
    elif option == "2":
        print(f"There are {len(shopping_list)}items in the list.")
        ind = int(input("enter item index: "))
        if 0 >= ind < len(shopping_list):
            shopping_list.pop(ind)
        print(f"your list is: {shopping_list}")
    elif option == "3":
        print("The following items remain in the list:")
        print(shopping_list)
        break
    else:
        print("invalid input")



prices = [10, 14, 22, 33, 44, 13, 22, 55, 66, 77]
totalsum = 0
while True:
    product = int(input("Please select product (1-10) 0 to Quit: "))
    if product == 0:
        break
    elif 1 <= product <= 10:
        price = prices[product - 1]
        totalsum += price
        print(f"Product: {product} Price: {price}")
    else:
        print("Invalid selection.")
print(f"Total: {totalsum}")
payment = int(input("Payment: "))
change = payment - totalsum
print(f"Change: {change}")

