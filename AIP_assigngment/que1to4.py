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
        item = input("enter item: ")
        shopping_list.pop(item)
    elif option == "3":
        break
    else:
        print("invalid input")
