#Calculate Profit or Loss

C_P = int(input('enter cost price:'))

S_P = int(input('enter selling price:'))

if S_P-C_P > 0:
    Prof = S_P-C_P
    print("profit amount is = : ", Prof)
elif S_P-C_P < 0:
    Loss = C_P-S_P
    print("loss amount is = :" ,Loss)
else:
    print("No profit No loss ")