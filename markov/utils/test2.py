with open("../data/example.fa") as f:
    data = [line.strip() for line in f]
new_data = "".join(data)
print(list(new_data))
