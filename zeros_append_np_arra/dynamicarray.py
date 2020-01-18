
import numpy as np

def fun(old,neww, rowsup,colsmov):
	b = old
	f = old.shape	
	b = np.append(neww,b,0)
	

	print(b)
	print("cutting\n")
	res  = b[0:f[0],:]

	colara = np.zeros(( f[0], abs(colsmov)))
	if(colsmov<0):
		res= np.append(colara,res,1)
		dim=res.shape
		
		colmax = dim[1]+colsmov
		res = res[0:f[0],0:(colmax)]
		print("wtf")
	else:
		res= np.append(res,colara,1)
		dim=res.shape
		res = res[0:f[0],colsmov:dim[1]]
	return res

oldx =2 
oldy =3
newx = 4
newy = 5
rowsup = newy - oldy
colsmov = newx-oldx


a= np.array([[1,2,3],[20,30,40],[39,49,59],[77,88,99]])
dim = a.shape
neww = np.zeros((rowsup,dim[1],))

print(a)
print("lmaoo\n")


g = fun(a,neww,rowsup,colsmov)


print(g)