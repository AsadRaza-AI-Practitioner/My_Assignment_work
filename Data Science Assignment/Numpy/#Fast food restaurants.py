#Fast food restaurants
import numpy as np

#city,country,latitude,longitude,name,postalCode = np.genfromtxt('FastFoodRestaurants.csv', delimiter=',', usecols=(1,2,4,5,6,7), unpack=True, dtype=('U66','U66',float,float,'U132',int), skip_header=1,invalid_raise=False )
city,country,latitude,longitude,name = np.genfromtxt('FastFoodRestaurants.csv', delimiter=',', usecols=(1,2,4,5,6), unpack=True, dtype=('U100','U100','f8','f8','U100'),encoding='utf-8',skip_header=1,invalid_raise=False )

print(city)
print(country)
print(latitude)
print(longitude)
print(name)
#Statistics_operation_on_fast_food_restaurant
cleaned_longitude = longitude[~np.isnan(longitude)]

print("Fast food restaurant latitude mean: " ,np.nanmean(longitude))
print("Fast food restaurant latitude average: " , np.nanmean(longitude)) #same as mean
print("Fast food restaurant latitude std: " , np.nanstd(longitude))
print("Fast food restaurant latitudemod: " , np.nanmedian(longitude))
print("Fast food restaurant latitude percentile - 3: " , np.nanpercentile(longitude,3))
print("Fast food restaurant latitudepercentile  - 3: " , np.nanpercentile(longitude,3))
print("Fast food restaurant latitude percentile  - 3: " , np.nanpercentile(longitude,3))
print("Fast food restaurant latitude min : " , np.nanmin(longitude))
print("Fast food restaurant latitude max : " , np.nanmax(longitude))
longitudeFiltered = longitude / np.nanmin(longitude)

# fast food restaurant math operations
print("fast food restaurant latitude square",np.square(longitude))
print("fast food restaurant latitude sqrt",np.sqrt(longitudeFiltered))
print("fast food restaurant latitude pow",np.power(longitudeFiltered,longitudeFiltered))
print("fast food restaurant latitude abs",np.abs(longitude))
# Perform basic arithmetic operations
addition=latitude+longitude
substraction=latitude-longitude
multiplication = latitude* longitude
division=latitude/longitude
print('fast food restaurant addition of latitude and longitude',addition)
print('fast food restaurant substraction of latitude and longitude',substraction)
print('fast food restaurant multiplication of latitude and longitude',multiplication)
print('fast food restaurant divison of latitudeand longitude',division)

#Trigonometric Functions

longitudePie = (longitude/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(longitudePie)
cosine_values = np.cos(longitudePie)
tangent_values = np.tan(longitudePie)
print("sine values are :",sine_values)
print("cosine values are :",cosine_values)
print("tangent values are :",tangent_values)
print("exponential values are:",np.exp(longitudePie))
# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(longitudePie)
log10_array = np.log10(longitudePie)
print("fast food restaurant longitude div natral logarithm value",log_array)
print("fast food restaurant longitude div base 10 logarithm value",log10_array)
