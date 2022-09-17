students = {}
for i in range(4):
    student_name = input()
    student_score = int(input())
    students[student_name] = student_score

for student_name in students.keys():
    student_score = students[student_name]
    grade = None

    if student_score >= 90:
        grade = 'A'
    elif student_score >= 80:
        grade = 'B'
    elif student_score >= 70:
        grade = 'C'
    elif student_score >= 60:
        grade = 'D'
    else:
        grade = 'F'
    print(f"{student_name}: {grade}")