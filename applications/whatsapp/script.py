import pandas as pd

with open("./deep-learning-tensorflow/applications/whatsapp/message.txt", "r") as file:
    content = file.read()

print(content)

contacts = pd.read_csv("./deep-learning-tensorflow/applications/whatsapp/contacts.csv")
# print(contacts.head())

for index, row in contacts.iterrows():
    name = row["Name"]
    number = "+" + str(row["Numbers"])
    print(f"Name: {name}, Phone Number: {number}")

print("WhatsApp application script executed successfully.")
