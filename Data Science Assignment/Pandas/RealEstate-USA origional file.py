#Usa real state state data
import pandas as pd
#  Read csv file to DataFrame
df= pd.read_csv('RealEstate-USA.csv',delimiter=",",parse_dates=[11],date_format={'prev_sold_date': '%d-%m-%Y'})
print(df)
#df.dtypes returns the data type of each column in your DataFrame.

#It tells pandas what kind of values are stored in each column.
print("df - data types" , df.dtypes)
# df.info() gives a summary of your entire DataFrame — showing:
#The number of rows and columns
#Column names
#How many non-null (filled) values are in each column
#The data type of each column
#Memory usage
print("df.info():   " , df.info() )
# display the last three rows
print('Last three Rows:')
print(df.tail(3))

# display the first three rows
print('First Three Rows:')
print(df.head(3))
print()
#Summary of Statistics of DataFrame using describe() method.
print("Summary of Statistics of DataFrame using describe() method", df.describe())

#Counting the rows and columns in DataFrame using shape(). It returns the no. of rows and columns enclosed in a tuple.
print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()
# access the Name column such as price
price=df['price']
print("access the Name column as price : df : ")
print(price)
print()
# access multiple columns
bed_acre_lot = df[['bed','acre_lot']]
print("access multiple columns: df : ")
print(bed_acre_lot)
print()

"""There are four primary ways to select rows with .loc. These include:
Selecting a single row
Selecting multiple rows
Selecting a slice of rows
Conditional row selection"""
# Case 1 : using .loc - default case - starts here
"""
Syntax               df.loc[row_indexer, column_indexer]              df.iloc[row_indexer, column_indexer]
Indexing Method      Label-based                                      Position-based indexing
Used for Reference   Row and column labels (names)                    Numerical indices of rows and columns (starting from 0)
"""
#Selecting a single row using .loc
second_row = df.loc[1]
print("#Selecting a single row using .loc")
print(second_row)
print()

#Selecting multiple rows using .loc
multiples_rows = df.loc[[1, 3]]
print("#Selecting multiple rows using .loc")
print(multiples_rows)
print()

#Selecting a slice of rows using .loc
slice_of_second_row = df.loc[1:5]
print("#Selecting a slice of rows using .loc")
print(slice_of_second_row)
print()

#Conditional selection of rows using .loc
#in this case we are  applying a condition where in city filter that in which Ciales belong to city column
second_row8 = df.loc[df['city'] == 'Ciales']
print("#Conditional selection of rows using .loc")
print(second_row8)
print()

#Selecting a single column using .loc
second_row5 = df.loc[:5,'city']
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:5,['city','state']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

# Case 1 : using .loc - default case - ends here

#Selecting a slice of columns using .loc
second_row7 = df.loc[:5,'street':'house_size']
print("#Selecting a slice of columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df.loc[df['city'] == 'Ciales','street':'house_size']
print("#Combined row and column selection using .loc")
print(second_row8)
print()

# Case 3 : Using .iloc - starts here
"""Using .iloc: Selection by Integer Position
.iloc selects by position instead of label. This is the standard syntax of using .iloc: df.iloc[row_indexer, column_indexer]. There are two special things to look out for:

Counting starting at 0: The first row and column have the index 0, the second one index 1, etc.
Exclusivity of range end value: When using a slice, the row or column specified behind the colon is not included in the selection."""

#Selecting a single row using .iloc
second_row = df.iloc[0]
print("#Selecting a single row using .iloc")
print(second_row)
print()

#Selecting multiple rows using .iloc
second_row2 = df.iloc[[1, 3,5]]
print("#Selecting multiple rows using .iloc")
print(second_row2)
print()

#Selecting a slice of rows using .iloc
second_row3 = df.iloc[2:5]
print("#Selecting a slice of rows using .iloc")
print(second_row3)
print()

#Selecting a single column using .iloc
second_row5 = df.iloc[:,2]
print("#Selecting a single column using .iloc")
print(second_row5)
print()

#Selecting multiple columns using .iloc
second_row6 = df.iloc[:,[2,4]]
print("#Selecting multiple columns using .iloc")
print(second_row6)
print()

#Selecting a slice of columns using .iloc
second_row7 = df.iloc[:,2:4]
print("#Selecting a slice of columns using .iloc")
print(second_row7)
print()

#Combined row and column selection using .iloc
second_row8 = df.iloc[[1, 3,5],2:4]
print("#Combined row and column selection using .iloc")
print(second_row8)
print()

# Case 3 : Using .iloc - ends here

""""Pandas DataFrame Manipulation
DataFrame manipulation in Pandas involves editing and modifying existing DataFrames. Some common DataFrame manipulation operations are:

Adding rows/columns
Removing rows/columns
Renaming rows/columns"""

#Remove Rows/Columns from a Pandas DataFrame
#row deletion of 1 index
df.drop(1,axis=0,inplace=True)
# delete row with index 1
df.drop(2,axis=0,inplace=True)
# delete rows with index 4 and 5
df.drop([4,5],axis=0,inplace=True)
# display the modified DataFrame after deleting rows
print("modified dataframe is=:",df)

#column deletion
#delete state column
df.drop('state',axis=1,inplace=True)
#delete zip code column
df.drop('zip_code',axis=1, inplace=True)
#delete house_size,prev_sold_date
df.drop(['house_size','prev_sold_date'],axis=1, inplace=True)
# display the modified DataFrame after deleting column
print("modified dataframe after deletion of column",df)

#Rename Labels in a DataFrame
# rename column 'Name' to 'First_Name'
df.rename(columns={'status':'status_changed'},inplace=True)
#rename using mapper
df.rename(mapper={'stree':'street_changed', 'city':'city_renamed'},axis=1, inplace=True)
#data frame look after renaming
print('data frame after renaming', df)


#rename row label
df.rename(index={0:7},inplace=True)
#rename multiplr row indexing
df.rename(mapper={1:13,2:15},axis=0,inplace=True)
#modified dataframe after renaming of rows
print("modified dataframe after renaming of rows",df)


#Query method for searching anything in dataframe
query_row=df.query('price>145000')
print(query_row)
print(query_row.to_string())
print(len(query_row))


""""Pandas Data Cleaning
Data cleaning means fixing and organizing messy data. Pandas offers a wide range of tools and functions to help us clean and preprocess our data effectively.
"""
cleaned_df=df.dropna()
print('cleaned_df', cleaned_df)

# filling NaN values with 0
df.fillna(0, inplace=True)
print('Nan in data has been filled with 0', df)



