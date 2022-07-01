import pickle

print("start")
with open('letters_info.pkl', 'rb') as file:
    letter_info = pickle.load(file)

print(type(letter_info))
print(letter_info)