def get_students(N):
    students = {}
    for i in range(N):
        student_name = input()
        student_score = int(input())
        students[student_name] = student_score   
    return students
    
def get_grade(student_score):
    if student_score >=90:
        return "A"
    elif student_score >= 80:
        return "B"
    elif student_score >= 70:
        return "C"
    elif student_score >= 60:
        return "D"
    else:
        return "F"


N = int(input())
students = get_students(N)
for student_name in students.keys():
    print(f"{student_name}: {get_grade(students[student_name])}")