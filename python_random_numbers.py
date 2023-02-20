import numpy as np
from math import log2
from itertools import chain, product
import pandas as pd 
import sys

def pattern_counter(returns,n):
    base = 5
    columns = (base)**(n)
   
    #create a matrix of L rows and (L^n) columns depending how many days backward we're looking
    #n = 1, 2, 3, 4, 5, 6, 7 or 8
    pattern_matrix  = np.zeros((base,columns), dtype = int)
    
    indices = np.arange(n)
   
    for i in range(0,(returns.size-n)):
        n_array = np.take(returns,indices)

        #convert from the given base to decimal
        num = n_array.dot((base)**np.arange(n_array.size)[::-1])

        pattern_matrix[returns[n]][num]+=1
        indices = np.add(indices,1)
        n=n+1

    return pattern_matrix 


#H(Y) input entropy calculation
def input_entropy(arr):
    n_arr = arr / np.sum(arr)
    ent = 0
 
    for i in range(arr[0].size):
        #sum of probabilites of the columns 
        column = np.sum(n_arr[:,[i]])
        if(column > 0):
            ent = ent + (-1*column*log2(column)) #Shannon entropy  

    return ent 

#H(X) output entropy 
def output_entropy(arr):
    n_arr = arr / np.sum(arr)
    ent = 0 
    #sum of all rows
    sum_rows = np.sum(n_arr,axis=1)

    for i in range(sum_rows.size):
        if(sum_rows[i] > 0):
            ent = ent + (-1*sum_rows[i]*log2(sum_rows[i])) #Shannon Entropy 

    return ent 




#H(X,Y) joint entropy calculation 
def joint_entropy(arr):
    ent = 0
    n_arr = arr / np.sum(arr)
    nr_rows = np.shape(n_arr)[0]
    for i in range(nr_rows):
        for j in range(n_arr[0].size):
            if(n_arr[i][j] > 0):
                ent = ent + (-1*n_arr[i][j]*log2(n_arr[i][j])) #Shanon entropy 

        

    return ent

"""
returns = np.loadtxt(fname="XOM.txt",dtype = int)

    
random_returns = np.random.choice([0, 1], p=[0.5, 0.5], size=(returns.size)) 


non_random = np.zeros(8,dtype=float)
random = np.zeros(8,dtype=float)

    
for i in range(1,9):
     #print("\nDays prior " + str(i))
     prior = pattern_counter(returns,i)

     ent = joint_entropy(prior)
     #print("\nH(X,Y) =  " + str(ent))

     ent_1 = input_entropy(prior)
     #print("\nH(Y)   =  " + str(ent_1))

     ent_2 = output_entropy(prior)
     #print("\nH(X)   =  " + str(ent_2))
     #print("\nH(X|Y) =  " + str(ent - ent_1))
     non_random[i-1] = (ent - ent_1)

       

     prior_random = pattern_counter(random_returns,i)
     ent_random = joint_entropy(prior_random)
     ent_random_1 = input_entropy(prior_random)
     ent_random_2 = output_entropy(prior_random)
     random[i-1]=(ent_random - ent_random_1)

    #print(non_random)

    #print(random)

difference = np.zeros(8,dtype=float)
for i in range(0,8):
    difference[i] = (random[i] - non_random[i])

    #print(difference)
add = 0
for i in range(difference.size):
    if(difference[i] > 0):
        add = add + (difference.size - i)*difference[i]

result = 1 - add/36
    

print("THe result is: ")
print(result)

"""








#if you want to print the whole numpy array
#np.set_printoptions(threshold=sys.maxsize)


#adj_close = np.loadtxt(fname="googleADJClose.txt",dtype = float)
    #in x is stored the difference between adj_close values 
    #example adj_close[1] - adj_close[0]
    #or today - yesterday in terms of stock values 
#x = np.diff(adj_close)
#returns = np.where(x>0, 1, 0) 

#data_file = np.loadtxt(fname="VZint.txt",dtype = float)

#range_values = np.array([-1,-0.15229,-0.05569,0.04091,0.137508,1])

#bins the values in according to the range_values
#array1 = np.digitize(data_file ,range_values)
#returns = array1 - 1
returns= np.loadtxt(fname="VZ range.txt",dtype = int)

#returns = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2], size=(returns1.size)) 

f = open("AA results.txt","w+")
#for i in range(returns.size):
#    f.write(str(returns[i]) + "\n")

#f.close()



for i in range(1,9):
        print("\nDays prior " + str(i))
        f.write("\nDays prior " + str(i) + "\n")
        prior = pattern_counter(returns,i)

        ent = joint_entropy(prior)
        print("\nH(X,Y) =  " + str(ent))
        f.write(str(ent) + "\n\n")

        ent_1 = input_entropy(prior)
        print("\nH(Y)   =  " + str(ent_1))
        f.write(str(ent_1) + "\n\n")

        ent_2 = output_entropy(prior)
        print("\nH(X)   =  " + str(ent_2))
        print("\nH(X|Y) =  " + str(ent - ent_1))

        f.write(str(ent_2) + "\n\n")
        f.write(str(ent - ent_1) + "\n\n")


f.close()













 