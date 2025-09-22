#Distribute Items Equally - You have n candies and k students and candies left.

n = int(input("number of candies is=:"))

k = int(input("number of students is =:"))

candies = n % k

if candies==0:

    print ("candies are divided equally amomg each student")

else:
    print ("candies are not  divided equally amomg each student")
