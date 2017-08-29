import pysol
import target_process

class ogd:

	def __init__(self,**params):
		self.params=params

	
	def fit(self,X,Y):
		[Y_new,self.old_to_new,self.new_to_old,num_class]=target_process.translate(Y)
		self.classifier=pysol.SOL('ogd',num_class,**self.params)
		self.classifier.fit(X,Y_new)
	
	def predict(self, X):
		return self.classifier.predict(X)
