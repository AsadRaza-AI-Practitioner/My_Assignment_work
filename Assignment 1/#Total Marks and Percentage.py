#Total Marks and Percentage

Eng = int(input("english marks aout of 100 is: "))
maths = int(input("maths marks out of 100 is: "))
sci = int(input(" sci marks out of 100 is : "))
urdu = int(input("urdu marks out of 100 is : "))
isl = int(input("islamiat marks out of 100 is : "))

Obtained_marks = Eng+maths+sci+urdu+isl
Avg_marks = Obtained_marks/5

print(type(Avg_marks))

percentage = (Obtained_marks*100)/500

print(type(percentage))

print('percentage is = ', percentage)