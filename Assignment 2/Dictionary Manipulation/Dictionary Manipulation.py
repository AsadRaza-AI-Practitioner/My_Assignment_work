#Dictionary Manipulation

#Check if a value exists in a dictionary

a={'b':'busy','c':'class','d':'door'}

e= 'class'
if e in a.values():
    print("checked value exist in my taken dictionary")
else:
    print("checked value doest exist in dictionary") 


#Get the key of a minimum value from the following dictionary

a={'b':'busy','c':'class','d':'door','e': 'enter'}

print(min(a,key=a.get))
print(max(a,key=a.get))

#Delete a list of keys from a dictionary

D={'b':'busy','c':'class','d':'door','e': 'enter'}
rk=['d','e']
for i in D if rk in d:
    del [D(rk)]

print(D)






















