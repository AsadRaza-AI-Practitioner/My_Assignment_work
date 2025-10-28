#
import numpy as np
brokered_by,status,price,bed_bath,acre_lot,	street,	city,state,	zip_code,house_size=np.genfromtxt('RealEstate-USA.csv',delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10),unpack=True, dtype=None,skip_header=1,invalid_raise=False)


print(status)
print(status)
print(price)
print(bed_bath)
print(acre_lot)
print(street)
print(city)
print(state)
print(zip_code)
# Usa real state _statistics operations
print("usa real state price mean",np.mean(price))
print("usa real state price average",np.average(price))
print("use real state price standard",np.std(price))
print("use real state price median_9",np.percentile(price,9))
print("use real state price median_25",np.percentile(price,25))
print("use real state price min",np.min(price))
print("use real state price max",np.max(price))

# Usa real state _maths operations
print("usa real state price square",np.square(price))
print("usa real state price square root",np.sqrt(price))
print("usa real state price power",np.pow(price,2))
print("usa real state abs", np.abs(price))

#Perform basic arithmetic operations
addition=price+bed_bath
substraction=price-bed_bath
multiplicaion=price*bed_bath
division=price/bed_bath
print("addition operation on price ",addition)
print("substraction operation on price ",substraction)
print("multiplication operation on price ",multiplicaion)
print("division operation on price ",division)
#trignometric
pricePie = (price/np.pi) +1
sine_of_price=np.sin(pricePie)
cos_of_price=np.cos(pricePie)
tan_of_price=np.tan(pricePie)
print("sine=",sine_of_price)
print("cosine=",cos_of_price)

# Calculate the natural logarithm and base-10 logarithm
log_array=np.log(pricePie)
log_10array=np.log10(pricePie)
print("natural log values=",log_array)
print('base_10 log values=',log_10array)

#Zameen.com price Plus bed_bath - 2 dimentional arrary
d2price_bed_bath=np.array([price,bed_bath])
print("usa real state  2D array values",d2price_bed_bath)
# check the dimension of array1
print("usa real state  2D array values_ dimension",d2price_bed_bath.ndim)
# return total number of elements in array1
print("usa real state  2D array values_ total number of elements",d2price_bed_bath.size)
# return a tuple that gives size of array in each dimension
print("usa real state  2D array values_ gives size of array in each dimension",d2price_bed_bath.shape)
# check the data type of array1
print("usa real state  2D array values_ data type of array1",d2price_bed_bath.dtype)
d2price_bed_bath_slice=d2price_bed_bath[1:,:5]

print("slicing_array=",d2price_bed_bath_slice)
d2price_bed_bath_slice1=d2price_bed_bath[2:5:2,4:5:2]

print("slicing_array=",d2price_bed_bath_slice1)
#indexing sliced array
d2price_bed_bath_slice_index=d2price_bed_bath_slice[0,2]
print("indexing of array is ",d2price_bed_bath_slice_index)
#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(d2price_bed_bath):
    print(elem)
#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(d2price_bed_bath):
    print(index, elem)

    """# for loop
rows = np.shape(d2price_bed_bath[0])[0]
cols = np.shape(d2price_bed_bath[1])[0]
for i in range(0, (rows + 1)):
    for j in range(0, (cols + 1)):
        print (d2price_bed_bath[i,j])
"""
# 2 x 149 ========>>>>> 1  x 298 - reshape
d2price_bed_bath4TO100 = np.reshape(d2price_bed_bath, (4, 100))
print("us real state price Plus bed_bath - 2 dimentional arrary - np.reshape(d2price_bed_bath, (4, 100)) : " , d2price_bed_bath4TO100)
print("Zameen.com Long Plus Lat - 2 dimentional arrary - np.reshape(d2price_bed_bath, (4, 100)) : Size " , d2price_bed_bath4TO100.size)
print("Zameen.com Long Plus Lat - 2 dimentional arrary - np.reshape(d2price_bed_bath, (4, 100)) : ndim " , d2price_bed_bath4TO100.ndim)
print("Zameen.com Long Plus Lat - 2 dimentional arrary - np.reshape(d2price_bed_bath, (4, 100)) : shape " , d2price_bed_bath4TO100.shape)
print("Zameen.com Long Plus Lat - 2 dimentional arrary - np.reshape(d2price_bed_bath, (4, 100)) : ndim " , d2price_bed_bath4TO100.ndim)




print()
