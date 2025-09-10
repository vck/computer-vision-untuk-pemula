"""
Python 101 for Computer Vision - Script Version
This script covers the basics of Python programming needed for computer vision.
"""

# 1. Variables and Data Types
print("=== Variables and Data Types ===")
age = 25
height = 5.9
name = "Alice"
is_student = True

print(f"Age: {age}, Type: {type(age)}")
print(f"Height: {height}, Type: {type(height)}")
print(f"Name: {name}, Type: {type(name)}")
print(f"Is Student: {is_student}, Type: {type(is_student)}")

# 2. Basic Operators
print("\n=== Basic Operators ===")
a = 10
b = 3

print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Integer Division: {a} // {b} = {a // b}")
print(f"Modulus: {a} % {b} = {a % b}")
print(f"Exponentiation: {a} ** {b} = {a ** b}")

# 3. Control Structures
print("\n=== Control Structures ===")
temperature = 25

if temperature > 30:
    print("It's hot outside!")
elif temperature > 20:
    print("It's a pleasant day.")
else:
    print("It's cold outside!")

# Loops
print("Counting from 1 to 5:")
for i in range(1, 6):
    print(i)

# 4. Functions
print("\n=== Functions ===")

def greet(name):
    return f"Hello, {name}!"

def calculate_area(length, width=1):
    return length * width

print(greet("Alice"))
print(f"Area (5x3): {calculate_area(5, 3)}")
print(f"Area (5x1): {calculate_area(5)}")

# 5. Data Structures
print("\n=== Data Structures ===")

# Lists
fruits = ["apple", "banana", "orange", "grape"]
print(f"Fruits: {fruits}")
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")

fruits.append("mango")
fruits.remove("banana")
print(f"After modifications: {fruits}")
print(f"Number of fruits: {len(fruits)}")

# Dictionaries
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8
}

print(f"Student name: {student['name']}")
print(f"Student details: {student}")

student["graduation_year"] = 2024
print(f"Updated student info: {student}")

print(f"Keys: {list(student.keys())}")
print(f"Values: {list(student.values())}")

# 6. File Handling
print("\n=== File Handling ===")

# Writing to a file
with open("sample.txt", "w") as file:
    file.write("This is a sample text file.\n")
    file.write("It contains multiple lines.\n")
    file.write("Perfect for learning file handling in Python!\n")

print("File 'sample.txt' has been created.")

# Reading from a file
with open("sample.txt", "r") as file:
    content = file.read()
    print("File content:")
    print(content)

# 7. Introduction to NumPy
print("\n=== Introduction to NumPy ===")

try:
    import numpy as np
    
    # Creating arrays
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    
    print(f"1D Array: {arr1}")
    print(f"2D Array:\n{arr2}")
    print(f"Shape of 2D array: {arr2.shape}")
    
    # Array operations
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    print(f"Addition: {a + b}")
    print(f"Multiplication: {a * b}")
    print(f"Dot product: {np.dot(a, b)}")
    
    # Creating special arrays
    zeros = np.zeros((3, 3))
    ones = np.ones((2, 4))
    identity = np.eye(3)
    
    print(f"Zeros array:\n{zeros}\n")
    print(f"Ones array:\n{ones}\n")
    print(f"Identity matrix:\n{identity}")
    
except ImportError:
    print("NumPy is not installed. Please install it using: pip install numpy")

# 8. Introduction to Matplotlib
print("\n=== Introduction to Matplotlib ===")

try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Matplotlib is available for creating visualizations.")
    print("To see plots, run the Jupyter notebook version of this tutorial.")
    
except ImportError:
    print("Matplotlib is not installed. Please install it using: pip install matplotlib")

print("\n=== Practice Exercises ===")
print("Try these exercises to reinforce your learning:")
print("1. Write a function that takes a list of numbers and returns the average.")
print("2. Create a dictionary to store information about your favorite movies.")
print("3. Use NumPy to create a 5x5 matrix filled with random numbers.")
print("4. Plot a histogram of 1000 random numbers from a normal distribution.")
print("5. Write a program that reads a text file and counts word frequency.")

print("\n=== Summary ===")
print("You've learned Python basics for computer vision:")
print("- Variables, data types, and operators")
print("- Control structures (conditionals, loops)")
print("- Functions and data structures")
print("- File handling")
print("- NumPy for numerical computing")
print("- Matplotlib for visualization")