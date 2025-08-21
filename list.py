names = ["Misra", "Nirmala" , "misu", "dhana" , "niru"]
names.append("Abhay")
names.remove("niru")
names.insert(4,"paras")

otherNames = ["md", "laxmi"]
names.extend(otherNames)
what_index=names.index("misu")
if "dhana" in names:
    "dhana found"
names.sort()
print(names)