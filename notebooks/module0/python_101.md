# Python 101 for Computer Vision

Welcome to the foundational Python programming module for computer vision! This module is designed for beginners who want to learn Python specifically for computer vision applications.

## What You'll Learn

In this module, we'll cover the essential Python concepts you need to understand before diving into computer vision:

1. Python basics (variables, data types, operators)
2. Control structures (conditionals, loops)
3. Functions and modules
4. Data structures (lists, dictionaries, tuples)
5. File handling
6. Introduction to NumPy for numerical computing
7. Introduction to Matplotlib for visualization

## Prerequisites

No prior programming experience is required, but basic computer literacy is assumed.

## 1. Python Basics

### 1.1 Variables and Data Types

In Python, you don't need to declare variable types explicitly. Python automatically determines the type based on the value assigned.

```python
# Integer
age = 25
print(f"Age: {age}, Type: {type(age)}")

# Float
height = 5.9
print(f"Height: {height}, Type: {type(height)}")

# String
name = "Alice"
print(f"Name: {name}, Type: {type(name)}")

# Boolean
is_student = True
print(f"Is Student: {is_student}, Type: {type(is_student)}")
```

### 1.2 Basic Operators

Python supports various operators for mathematical and logical operations.

```python
# Arithmetic operators
a = 10
b = 3

print(f"Addition: {a} + {b} = {a + b}")
print(f"Subtraction: {a} - {b} = {a - b}")
print(f"Multiplication: {a} * {b} = {a * b}")
print(f"Division: {a} / {b} = {a / b}")
print(f"Integer Division: {a} // {b} = {a // b}")
print(f"Modulus: {a} % {b} = {a % b}")
print(f"Exponentiation: {a} ** {b} = {a ** b}")
```

```python
# Comparison operators
x = 5
y = 10

print(f"{x} == {y}: {x == y}")
print(f"{x} != {y}: {x != y}")
print(f"{x} > {y}: {x > y}")
print(f"{x} < {y}: {x < y}")
print(f"{x} >= {y}: {x >= y}")
print(f"{x} <= {y}: {x <= y}")
```

```python
# Logical operators
p = True
q = False

print(f"{p} and {q}: {p and q}")
print(f"{p} or {q}: {p or q}")
print(f"not {p}: {not p}")
```

## 2. Control Structures

### 2.1 Conditionals

Conditional statements allow you to execute different code paths based on certain conditions.

```python
temperature = 25

if temperature > 30:
    print("It's hot outside!")
elif temperature > 20:
    print("It's a pleasant day.")
else:
    print("It's cold outside!")
```

### 2.2 Loops

Loops allow you to repeat code multiple times.

```python
# For loop
print("Counting from 1 to 5:")
for i in range(1, 6):
    print(i)
```

```python
# While loop
count = 0
print("Counting with while loop:")
while count < 5:
    count += 1
    print(count)
```

## 3. Functions

Functions are reusable blocks of code that perform specific tasks.

```python
# Simple function
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
```

```python
# Function with default parameter
def calculate_area(length, width=1):
    return length * width

print(f"Area (5x3): {calculate_area(5, 3)}")
print(f"Area (5x1): {calculate_area(5)}")  # Uses default width of 1
```

## 4. Data Structures

### 4.1 Lists

Lists are ordered collections that can contain different types of elements.

```python
# Creating and accessing lists
fruits = ["apple", "banana", "orange", "grape"]
print(f"Fruits: {fruits}")
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")
```

```python
# List operations
fruits.append("mango")
print(f"After adding mango: {fruits}")

fruits.remove("banana")
print(f"After removing banana: {fruits}")

print(f"Number of fruits: {len(fruits)}")
```

### 4.2 Dictionaries

Dictionaries store key-value pairs and are useful for organizing data.

```python
# Creating and accessing dictionaries
student = {
    "name": "Alice",
    "age": 20,
    "major": "Computer Science",
    "gpa": 3.8
}

print(f"Student name: {student['name']}")
print(f"Student details: {student}")
```

```python
# Dictionary operations
student["graduation_year"] = 2024
print(f"Updated student info: {student}")

print(f"Keys: {list(student.keys())}")
print(f"Values: {list(student.values())}")
```

## 5. File Handling

Working with files is essential for loading and saving data in computer vision applications.

```python
# Writing to a file
with open("sample.txt", "w") as file:
    file.write("This is a sample text file.\n")
    file.write("It contains multiple lines.\n")
    file.write("Perfect for learning file handling in Python!\n")

print("File 'sample.txt' has been created.")
```

```python
# Reading from a file
with open("sample.txt", "r") as file:
    content = file.read()
    print("File content:")
    print(content)
```

## 6. Introduction to NumPy

NumPy is a fundamental library for scientific computing in Python, especially important for computer vision.

```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print(f"1D Array: {arr1}")
print(f"2D Array:\n{arr2}")
print(f"Shape of 2D array: {arr2.shape}")
```

```python
# Array operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")
print(f"Dot product: {np.dot(a, b)}")
```

```python
# Creating special arrays
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
identity = np.eye(3)

print(f"Zeros array:\n{zeros}\n")
print(f"Ones array:\n{ones}\n")
print(f"Identity matrix:\n{identity}")
```

## 7. Introduction to Matplotlib

Matplotlib is used for creating visualizations, which is important for understanding image data.

```python
import matplotlib.pyplot as plt
import numpy as np

# Simple line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple Plot')
plt.legend()
plt.grid(True)
plt.show()
```

```python
# Bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(8, 5))
plt.bar(categories, values, color='skyblue')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')
plt.show()
```

## 8. Practice Exercises

Try these exercises to reinforce your learning:

1. Write a function that takes a list of numbers and returns the average.
2. Create a dictionary to store information about your favorite movies (title, year, rating).
3. Use NumPy to create a 5x5 matrix filled with random numbers between 0 and 1.
4. Plot a histogram of 1000 random numbers generated from a normal distribution.
5. Write a program that reads a text file and counts the frequency of each word.

## Summary

In this module, you've learned the fundamental Python concepts needed for computer vision:

- Basic data types and operators
- Control structures (conditionals and loops)
- Functions for code reusability
- Essential data structures (lists, dictionaries)
- File handling for data I/O
- NumPy for numerical computing
- Matplotlib for data visualization

These foundations will serve you well as you progress through the computer vision modules. In the next module, we'll dive into the fundamentals of computer vision itself!