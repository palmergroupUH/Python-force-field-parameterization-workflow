import numpy as np 

# gradient-free optimizer: Nelder-Mead simplex method:  

class NelderMeadSimplex:
	
	def __init__(self,input,objective,Output=None):

		self.vertices = parameters

		self.cost = obj
	
		self.nvertices = NVertices 

		self.nparameter = NVertices - 1  

		if ( input_para ) : 

			self.input_para = self.input_para 

		if ( simulation ) : 

			self.simulation = simulation 
		
		if ( costfunction ): 
		
			self.costfunction = costfunction 
		
		if ( output ) : 

			self.output = output  

	def TransformationCoeff(self,keyword): 	

		if ( keyword == "standard" ): 

			alpha = 1 ; kai =  2.0 ; gamma = 0.5 ; sigma = 0.5 

			return alpha , kai,  gamma , sigma
			
		elif ( keyword == "adaptive" ): 

			alpha = 1  ; kai =  1 + 2.0/self.nparameter  ; gamma = 0.75-1.0/(2*self.nparameter)  ; sigma = 1-1.0/self.nparameter 

		return alpha,kai, gamma, sigma 

	def Constrain(self):  

	def Evaluate_Vertices(): 

		self.vertices = costfunc 

		self.Sort() 	
		
		return None 

	def Sort(self): 
		
		low_to_high = np.argsort(self.cost) 

		self.vertices = self.vertices[low_to_high,:]
		
		self.cost = self.cost[low_to_high]

		self.worstvertex = self.vertices[-1,:]

		self.secondworstvertex = self.vertices[-2,:] 

		self.bestvertex = self.vertices[0,:]

	def Centroid(self):  

		except_worst = self.vertices[:-1,]	

		self.centroid = np.sum(except_worst,axis=0)/( self.vertices[:,0].size-1) 
			
		return self.centroid 
		
	def Reflect(self): 

		self.reflected = self.centroid + alpha*(self.centroid - self.worstvertex) 

		return self.reflected  

	def Accept(self,vertice):  


	def Expand(self):  

		self.expanded = self.centroid + kai*(self.reflected - self.centroid ) 

		return self.expanded  

	def Outside_Contract(self):  

		self.outsidecontracted = self.centroid + gamma*(self.reflected - self.centroid)
		
		return self.outsidecontracted  
		
	def Inside_Constract(self): 	

		self.insidecontracted = self.centroid + gamma*( self.worstvertex - self.centroid)
	
		return self.insidecontracted 
	
	def Shrink(self):   

		self.shrinkvertices = np.zeros(( nvertices-1,nvertices-1) )

		for i in range(self.vertices-1): 
	
			self.shrinkvertices[i,:] = self.bestvertex + sigma(self.vertices[i+1,:]- self.bestvertex) 	

		return self.shrinkvertices 

	def Run(self):  

		for itera in range(self.max_iteration):  
		
			simplex.order() 	

			simplex.centroid() 

			# Reflection 

			fval_reflect = simplex.Reflect() 

			if ( simplex.best <= fval_reflect < simplex.lousy ): 

				simplex.Accept_Reflect() 	

				continue 
			
			# Expansion
		
			if ( fval_reflect < simplex.best ): 

				fval_expand = simplex.Expand() 			

				if ( fval_expand < fval_reflect ):  	

					simplex.Accept_Expand() 

					continue 

				else:

					simplex.Accept_Reflect() 

					continue 
			
			# Contraction 	

			if ( fval_reflect >= simplex.lousy ): 	

				if ( simplex.lousy <= fval_reflect < simplex.worst ):  

					fval_oconstract = simplex.Outside_Contract() 

					if ( fval_oconstract <= fval_reflect ): 

						simplex.Accept_Outside_Contract() 

						continue 

					else:

						simplex.Shrink() 

						continue 

				if ( fval_reflect >= simplex.worst ): 

					fval_iconstract = simplex.Inside_Contract()	

					if ( fval_iconstract < simplex.worst ):  	

						simplex.Accept_Inside_Contract() 
		
						continue 
				
					else: 				

						simplex.Shrink()
							
						continue 	
					
	def Terminate(self,): 

		# check the variance of simplex vertices   
	
		# check the variance of functions at simplex vertices  

		# check the total number of variances 
		

		return None 


# gradient optimizer:    
