#Reverse a list in Python
a= ['asad',23,34,22]
a.reverse()
print("revered list:",a)

#Turn every item of a list into its square
b= [1,2,3,4,5,6]

square=[y**2 for y in b]

print(square)

#Remove empty strings from the list of strings

str_list= ['asad ','age','']
string= [i for i in  str_list if i.strip()]
print("string after removal od empty string is=",string)

#Add new item to list after a specified item

str_list.insert(1,'house')
print(str_list)

#Replace list’s item with new value if found

str_list[1]='lahore'
print(str_list)



