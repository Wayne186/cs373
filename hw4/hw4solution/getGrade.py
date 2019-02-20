import numpy as np
import scipy
import traceback


def test_1(dataX,datay,log):
	from greedysubset import run as gsrun
	import linreg
	import sol_greedysubset as sol_gs
	import numpy as np

	F = [1,2,5] # hard-coding F for 3 test cases
	
	
	t_case_scores = [3.0/3,3.0/3,3.0/3]
	totalX=len(dataX)
	vector_format = 0
	vector_format1 = 0
	score = 0.0
	deduct = False
	for t in range(0,totalX):
		try:
			X,y=dataX[t],datay[t]
			(n,dim) = np.shape(X)
			trainSize = int(n*4/5)

			S,thetaS = gsrun(F[t],X[0:trainSize],y[0:trainSize])
			S_sol,thetaS_sol = sol_gs.run(F[t],X[0:trainSize],y[0:trainSize])
			
			if isinstance(S, list):
				S = np.asarray(S)
				deduct = True
			if isinstance(thetaS, list):
				thetaS = np.asarray(S)
				deduct = True
			if isinstance(S,set):
				S = np.asarray(list(S))
				deduct = True
			if isinstance(S,np.matrix):
				S = np.asarray(S)

			if isinstance(S,np.ndarray):
				S = S.astype(int)

			okthetaS = True
			okS = True
			# try and correct dimension if in wrong format
			if (np.shape(thetaS) == (F[t],1)):
				vector_format1 = 1
			elif (np.shape(thetaS) == (1,F[t])):
				vector_format1 = 2
				thetaS=thetaS.reshape(F[t],1)
			elif (np.shape(thetaS)==(F[t],)):
				vector_format1 = 3
				thetaS = thetaS.reshape(F[t],1)
			else:
				okthetaS = False

			if (np.shape(S) == (F[t],1)):
				vector_format = 1
			elif (np.shape(S) == (1,F[t])):
				vector_format = 2
				S = S.reshape(F[t],1)
			elif (np.shape(S)==(F[t],)):
				vector_format = 3
				S = S.reshape(F[t],1)
			else:
				okS = False

			#check int values
			# same shape and same int vals
			if (okS and okthetaS):
				res1 = (len(np.intersect1d(S,S_sol))+0.0)/len(np.union1d(S,S_sol))
				# norm(y - Xtheta)/norm(y)
				#perfect or almost perfect linear regression
				a = y[trainSize:] - np.dot(X[trainSize:][:,S.reshape(len(S),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))            

				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2_tmp1 = 1.0
				else:
					res2_tmp1 = score_sol/score_hw

				a = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))            

				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2_tmp2 = 1.0
				else:
					res2_tmp2 = score_sol/score_hw

				res2 = max(res2_tmp1,res2_tmp2)
				q_score = ((res1 * t_case_scores[t])/2.0) + ((res2 * t_case_scores[t])/2.0)
				score += q_score
			elif (okS and not(okthetaS)):
				res1 = (len(np.intersect1d(S,S_sol))+0.0)/len(np.union1d(S,S_sol))
				q_score = ((res1 * t_case_scores[t])/2.0)
				score += q_score
			elif (okthetaS and not(okS)):
				a = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))
				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2 = 1.0
				else:
					res2 = score_sol/score_hw

				q_score = ((res2 * t_case_scores[t]))/2.0
				score += q_score

		except Exception as e:
			traceback.print_exc()
			log.write(traceback.format_exc())
			return 0,0,False

		
	if  vector_format == 1 and vector_format1 == 1:
		return score,1,deduct
	elif vector_format != 1:
		return score,vector_format,deduct
	elif vector_format1 != 1:
		return score,vector_format1,deduct

def test_2(dataX,datay,log):
	from forwardfitting import run as ffrun
	import linreg
	import sol_forwardfitting as sol_ff
	import numpy as np

	F = [1,2,5] # hard-coding F for 3 test cases
	
	
	t_case_scores = [3.0/3,3.0/3,3.0/3]
	totalX=len(dataX)
	vector_format = 0
	vector_format1 = 0
	score = 0.0
	deduct = False
	for t in range(0,totalX):
		try:
			X,y=dataX[t],datay[t]
			(n,dim) = np.shape(X)
			trainSize = int(n*4/5)

			S,thetaS = ffrun(F[t],X[0:trainSize],y[0:trainSize])
			S_sol,thetaS_sol = sol_ff.run(F[t],X[0:trainSize],y[0:trainSize])

			if isinstance(S, list):
				S = np.asarray(S)
				deduct = True
			if isinstance(thetaS, list):
				thetaS = np.asarray(S)
				deduct = True
			if isinstance(S,set):
				S = np.asarray(list(S))
				deduct = True
			if isinstance(S,np.matrix):
				S = np.asarray(S)

			if isinstance(S,np.ndarray):
				S = S.astype(int)

			okthetaS = True
			okS = True
			# try and correct dimension if in wrong format
			if (np.shape(thetaS) == (F[t],1)):
				vector_format1 = 1
			elif (np.shape(thetaS) == (1,F[t])):
				vector_format1 = 2
				thetaS=thetaS.reshape(F[t],1)
			elif (np.shape(thetaS)==(F[t],)):
				vector_format1 = 3
				thetaS = thetaS.reshape(F[t],1)
			else:
				okthetaS = False

			if (np.shape(S) == (F[t],1)):
				vector_format = 1
			elif (np.shape(S) == (1,F[t])):
				vector_format = 2
				S = S.reshape(F[t],1)
			elif (np.shape(S)==(F[t],)):
				vector_format = 3
				S = S.reshape(F[t],1)
			else:
				okS = False

			#check int values
			# same shape and same int vals

			if (okS and okthetaS):
				res1 = (len(np.intersect1d(S,S_sol))+0.0)/len(np.union1d(S,S_sol))
				# norm(y - Xtheta)/norm(y)
				#perfect or almost perfect linear regression
				a = y[trainSize:] - np.dot(X[trainSize:][:,S.reshape(len(S),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))            

				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2_tmp1 = 1.0
				else:
					res2_tmp1 = score_sol/score_hw

				a = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))            

				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2_tmp2 = 1.0
				else:
					res2_tmp2 = score_sol/score_hw

				res2 = max(res2_tmp1,res2_tmp2)
				q_score = ((res1 * t_case_scores[t])/2.0) + ((res2 * t_case_scores[t])/2.0)
				score += q_score
			elif (okS and not(okthetaS)):
				res1 = (len(np.intersect1d(S,S_sol))+0.0)/len(np.union1d(S,S_sol))
				q_score = ((res1 * t_case_scores[t])/2.0)
				score += q_score
			elif (okthetaS and not(okS)):
				a = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))
				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2 = 1.0
				else:
					res2 = score_sol/score_hw

				q_score = ((res2 * t_case_scores[t]))/2.0
				score += q_score

		except Exception as e:
			traceback.print_exc()
			log.write(traceback.format_exc())
			return 0,0,False

		
	if  vector_format == 1 and vector_format1 == 1:
		return score,1,deduct
	elif vector_format != 1:
		return score,vector_format,deduct
	elif vector_format1 != 1:
		return score,vector_format1,deduct

def test_3(dataX,datay,log):
	from myopicfitting import run as mfrun
	import linreg
	import sol_myopicfitting as sol_mf
	import numpy as np

	F = [1,2,5] # hard-coding F for 3 test cases
	
	t_case_scores = [4.0/3,4.0/3,4.0/3]
	totalX=len(dataX)
	vector_format = 0
	vector_format1 = 0
	score = 0.0
	deduct = False
	for t in range(0,totalX):
		try:
			X,y=dataX[t],datay[t]
			(n,dim) = np.shape(X)
			trainSize = int(n*4/5)

			S,thetaS = mfrun(F[t],X[0:trainSize],y[0:trainSize])
			S_sol,thetaS_sol = sol_mf.run(F[t],X[0:trainSize],y[0:trainSize])

			if isinstance(S, list):
				S = np.asarray(S)
				deduct = True
			if isinstance(thetaS, list):
				thetaS = np.asarray(S)
				deduct = True
			if isinstance(S,set):
				S = np.asarray(list(S))
				deduct = True
			if isinstance(S,np.matrix):
				S = np.asarray(S)

			if isinstance(S,np.ndarray):
				S = S.astype(int)

			okthetaS = True
			okS = True
			# try and correct dimension if in wrong format
			if (np.shape(thetaS) == (F[t],1)):
				vector_format1 = 1
			elif (np.shape(thetaS) == (1,F[t])):
				vector_format1 = 2
				thetaS=thetaS.reshape(F[t],1)
			elif (np.shape(thetaS)==(F[t],)):
				vector_format1 = 3
				thetaS = thetaS.reshape(F[t],1)
			else:
				okthetaS = False

			if (np.shape(S) == (F[t],1)):
				vector_format = 1
			elif (np.shape(S) == (1,F[t])):
				vector_format = 2
				S = S.reshape(F[t],1)
			elif (np.shape(S)==(F[t],)):
				vector_format = 3
				S = S.reshape(F[t],1)
			else:
				okS = False

			#check int values
			# same shape and same int vals

			if (okS and okthetaS):
				res1 = (len(np.intersect1d(S,S_sol))+0.0)/len(np.union1d(S,S_sol))
				# norm(y - Xtheta)/norm(y)
				#perfect or almost perfect linear regression
				a = y[trainSize:] - np.dot(X[trainSize:][:,S.reshape(len(S),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))            

				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2_tmp1 = 1.0
				else:
					res2_tmp1 = score_sol/score_hw

				a = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))            

				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2_tmp2 = 1.0
				else:
					res2_tmp2 = score_sol/score_hw

				res2 = max(res2_tmp1,res2_tmp2)
				q_score = ((res1 * t_case_scores[t])/2.0) + ((res2 * t_case_scores[t])/2.0)
				score += q_score
			elif (okS and not(okthetaS)):
				res1 = (len(np.intersect1d(S,S_sol))+0.0)/len(np.union1d(S,S_sol))
				q_score = ((res1 * t_case_scores[t])/2.0)
				score += q_score
			elif (okthetaS and not(okS)):
				a = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS)
				score_hw = (np.linalg.norm(a)/np.linalg.norm(y))
				b = y[trainSize:] - np.dot(X[trainSize:][:,S_sol.reshape(len(S_sol),)],thetaS_sol)
				score_sol = (np.linalg.norm(b)/np.linalg.norm(y))
				if ((abs(score_sol - score_hw) < 0.1) or (score_hw < score_sol)):
					res2 = 1.0
				else:
					res2 = score_sol/score_hw

				q_score = ((res2 * t_case_scores[t]))/2.0
				score += q_score

		except Exception as e:
			traceback.print_exc()
			log.write(traceback.format_exc())
			return 0,0,False

		
	if  vector_format == 1 and vector_format1 == 1:
		return score,1,deduct
	elif vector_format != 1:
		return score,vector_format,deduct
	elif vector_format1 != 1:
		return score,vector_format1,deduct

def gradeHW_4(stdID=""):
	import scipy.io
	print("------------------------------------------------------------")
	print("Grading "+stdID)
	
	X_1 = np.loadtxt("X_1.txt",delimiter=',')
	X_2 = np.loadtxt("X_2.txt",delimiter=',')
	X_3 = np.loadtxt("X_3.txt",delimiter=',')
	y_1 = np.loadtxt("y_1.txt",delimiter=',')
	y_2 = np.loadtxt("y_2.txt",delimiter=',')
	y_3 = np.loadtxt("y_3.txt",delimiter=',')
	y_1 = y_1.reshape((len(y_1),1))	
	y_2 = y_2.reshape((len(y_2),1))
	y_3 = y_3.reshape((len(y_3),1))





	dataX = [X_1,X_2,X_3]
	datay = [y_1,y_2,y_3]
	
	


	gradeFile = "hw4_grades.csv" 
	prob1,vector_format1=0,0
	prob2,vector_format2=0,0
	prob3,vector_format3=0,0	
	deduct1 = False
	deduct2 = False
	deduct3 = False
	with open(gradeFile,"a") as gf, open('log.txt',"a") as log:
		log.write("-----------------------------------"+"\n")
		log.write("Grading "+stdID+"\n")
		try:
			prob1,vector_format1,deduct1=test_1(dataX,datay,log)
			# pass
			if np.isnan(prob1):
				prob1 = 0
				deduct1 = False

		except Exception as e:
			traceback.print_exc()
			log.write(traceback.format_exc())
			pass
		try:
			prob2,vector_format2,deduct2=test_2(dataX,datay,log)
			# pass
			if np.isnan(prob2):
				prob2 = 0
				deduct2 = False
		except Exception as e:
			traceback.print_exc()
			log.write(traceback.format_exc())
			pass
		try:
			prob3,vector_format3,deduct3=test_3(dataX,datay,log)
			# pass
			if np.isnan(prob3):
				prob3 = 0
				deduct3 = False
		except Exception as e:
			traceback.print_exc()
			log.write(traceback.format_exc())
			pass
		final_score = (prob1+prob2+prob3)
		# deduct only once
		deducted = 0
		if (deduct1 or deduct2 or deduct3):
			final_score = final_score - 0.25
			deducted = 0.25

		final_score=str.format("%.1f"%final_score)
		print("Problem 1: %f\nVector Format : %d"%(prob1,vector_format1))
		print("Problem 2: %f\nVector Format : %d"%(prob2,vector_format2))
		print("Problem 3: %f\nVector Format : %d"%(prob3,vector_format3))
		print("Deducted: %f"%(deducted))
		print("Final Score: ",final_score)
		prob1=str.format("%.1f"%prob1)
		prob2=str.format("%.1f"%prob2)
		prob3=str.format("%.1f"%prob3)


		gf.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n"%(str(stdID), str(prob1),str(vector_format1),str(prob2),str(vector_format2),str(prob3),str(vector_format3),deducted,final_score))


if __name__ == "__main__":
	np.random.seed(26)
	import sys
	gradeFile = "hw4_grades.csv" 
	import os
	exists = os.path.isfile(gradeFile)
	if not os.path.isfile(gradeFile):
		with open(gradeFile,"a") as gf:
			gf.write("id,problem1,problem1_dim,problem2,problem2_dim,problem3,problem3_dim,deducted,score\n")
	if(len(sys.argv)==2):
		gradeHW_4(sys.argv[1])
	elif(len(sys.argv)==1):
		gradeHW_4()

