

gpa = [
    4.33,
    4.0,
    3.67,
    3.33,
    3.0,
]

grades = [
    3+3+6 + 2+2+3, # A+
    7+10 + 1+2+1, # A
    4+1 + 1, # A-
    1, # B+
    2, # B
]

apCredits = 10
totalCredits = 49

output = 0

for i in range(5):
    output += grades[i] * gpa[i]

unweighted = output / totalCredits

collegeGpa = (output - (grades[0] * 0.33))/totalCredits

output += apCredits * 0.5

weighted = output / totalCredits

print("UW 4.0 GPA: " + str(collegeGpa))
print("UW GPA: " + str(unweighted))
print("W GPA: " + str(weighted))
print("Weighted GPA History: ")
print("9th: 3.823529411764706")
print("10th: 3.949393939393939")
print("11th: 4.135102040816327")
