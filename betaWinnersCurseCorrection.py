version = 0.013
#################
Readme = '''
Given a set of --in summary stats and --pvalCutoff, script calculates Mean, MLE, and MSE beta estimators,
and returns the corrected data to --out file. Use --forceCutoff if summStats are unfiltered.

-v option allows for some simple diagnostic plotting.
To compare against a duplicate set of summary stats (ie. held out data for validation),
supply either file name with --otherIn, or if this was done previously, supply the 
--merged file to skip the lengthy snp alignment step. More diagnostic plotting will be returned here.

Example runs:
	To correct:
	python betaWinnersCurseCorrection.py --in <my_summary_stats.file> --out <output prefix>
	To compare to another file:
	python betaWinnersCurseCorrection.py --in <my_summary_stats.file> --out <output prefix> \
		--otherIn <my_replicate_summary_stats.file>
'''
#################
## open questions: is SE = sigma? Assumed YES

import sys
#import os
import matplotlib as mpl
#mpl.use('tkagg') #so show works, but not when run as a job. need agg for a subbed job.
mpl.use('Agg')
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import erf
from scipy.optimize import brent
from scipy.optimize import brentq
print("loaded packages")

# sys.argv = "pythonScriptName --in z040_in_bgen_SHR_all_halfGwas1_justSig.gwasOutput --otherIn \
# z030_out_bgen_SHR_all_halfGwas2.tab --out WCplots/z041_out_bgen_SHR_all_halfGwas_WCcorrected_justSig --forceCutoff \
# --merged z041_out_bgen_SHR_all_halfGwas_WCcorrected_justSig_withReplicate.txt --otherColNames .,chr,pos,.,A0,A1,.,.,.,.,beta,.,.,.,.,p-value".split(" ")


# sys.argv = "pythonScriptName --in z040_in_bgen_SHR_all_halfGwas1_justSigish.gwasOutput --otherIn \
# z030_out_bgen_SHR_all_halfGwas2.tab --out WCplots/z041_out_bgen_SHR_all_halfGwas_WCcorrected_justSigish --forceCutoff \
# --merged z041_out_bgen_SHR_all_halfGwas_WCcorrected_justSigish_withReplicate.txt --pvalCutoff 5e-4 --otherColNames .,chr,pos,.,A0,A1,.,.,.,.,beta,.,.,.,.,p-value".split(" ")
# print("TESTING ARGUMENTS USED ###################################################################################################")






####################################################### SETUP #######################################################

import argparse
parser = argparse.ArgumentParser(description="tldr: Correct summary stats for winners' curse.\n" + Readme, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--in', dest='summStats', type=str,
                   help='Summary stats file location.',required=True)
parser.add_argument('--out', dest='outPrefix', type=str,
                   help='Prune output file destination.',required=True)
parser.add_argument('--pvalCutoff', dest='gws_p', type=float, action='store', default=5e-8,
                   help="Cutoff, as a p-value. Will be converted to z-score (default: 5e-8).")
parser.add_argument('--otherIn', dest='summStats2', type=str,
                   help='Summary stats file location 2, for comparison. If provided, will be used to compare to original and plot.',required=False, default=None)
parser.add_argument('--merged', dest='summStatsMerged', type=str,
                   help='Merged output from providing --otherIn. If provided, merge will be skipped and file used instead.',required=False, default=None)
parser.add_argument('--forceCutoff', dest='forceCutoff', action='store_true',
                    help='Apply cutoff to summStats (default: use all snps in summStats).')
parser.add_argument('--colNames', dest='colNamesRaw', action='store', default = "rsid,chr,pos,A0,A1,beta,SE,p-value",
                    help='Comma delimited column names. Required columns: "beta,SE,p-value", and either "MarkerName" or "chr,pos,A0,A1" (default: "rsid,chr,pos,A0,A1,beta,SE,p-value").')
parser.add_argument('--otherColNames', dest='otherColNamesRaw', action='store', default = None,
                    help='Comma delimited column names. Required columns: "beta,p-value", and either "MarkerName" or "chr,pos,A0,A1" (default: Same as --colNames).')
parser.add_argument('--skipHeader', dest='skipHeader', action='store_true',
                    help='Skim summStats header (default: False).')
parser.add_argument('--quiet', '-q', action='count', default=0)
args = parser.parse_args()
if args.quiet == 0:
    print("WC_correction version %s" % version)
    print("args:")
    print(args)

colNames = args.colNamesRaw.split(",")
for name in ["beta","SE","p-value"]:
	if name not in colNames:
		raise Exception("Error: %s not in colnames: %s" % (name,colNames))

if ("MarkerName" not in colNames) and not (all([name in colNames for name in ["chr","pos","A0","A1"]])):
	raise Exception("Error: %s nor %s in colnames: %s" % ("MarkerName",["chr","pos","A0","A1"],colNames))

if args.otherColNamesRaw == None:
	args.otherColNamesRaw = args.colNamesRaw

otherColNames = args.otherColNamesRaw.split(",")
for name in ["beta","p-value"]:
	if name not in otherColNames:
		raise Exception("Error: %s not in colnames: %s" % (name,otherColNames))
if ("MarkerName" not in otherColNames) and not (all([name in otherColNames for name in ["chr","pos","A0","A1"]])):
	raise Exception("Error: %s nor %s in colnames: %s" % ("MarkerName",["chr","pos","A0","A1"],otherColNames))



# exit()
gws_p = args.gws_p
gws = abs(norm.ppf(gws_p))
magYlabel = "abs(Estimate) - signed(Repl_beta)"
#lets do some maths.


def magCalcRepl(row, colToCompare):
	return magCalc(row, colToCompare, "replicateBeta")

def magCalc(row, col1,col2):
	if row[col1] >= 0:
		return row[col1] - row[col2]
	else:
		return -row[col1] + row[col2]

def stdNormPdf(x):
	return (1.0/(2*np.pi)**(.5)) * np.e ** (-x**2/2)

def log_stdNormPdf(x): #checked to work
	return np.log((1.0/(2*np.pi)**(.5))) + -x**2/2

def stdNormCdf(x):
	return 1.0/2*(1 + erf(x/np.sqrt(2)))

# def inverseFunction(args, function=None): ### this is not working, unsure how to pass args to brent
# 	return -function(*args)

#log f(betaHat,beta)
#beta is true beta, betaHat is beta observed. cutoff is in z score units, sigma = SE
def log_f_betaHat_beta(beta, betaHat, sigma, cutoff):
	# p = 1/sigma * stdNormPdf((betaHat-beta)/cutoff)/(stdNormCdf(beta/sigma-cutoff) + stdNormCdf(-beta/sigma-cutoff))
	# log_p = np.log(p)
	#indicator fn not needed cuz only looking at sig loci
	log_p = log_stdNormPdf((betaHat-beta)/sigma) - np.log(sigma*(stdNormCdf(beta/sigma-cutoff) + stdNormCdf(-beta/sigma-cutoff)))
	return log_p

def Ef_betaHat_beta(beta, sigma, cutoff):
	expectedBetaHat = beta + sigma*(stdNormPdf(beta/sigma-cutoff) - stdNormPdf(-beta/sigma-cutoff))/(stdNormCdf(beta/sigma-cutoff) + stdNormCdf(-beta/sigma-cutoff))
	return expectedBetaHat

def findMLE(betaHat_obs, standardError, cutoff):
	#argmax(Beta) log f(BetaHat; Beta)
	sigma = standardError
	def minus_log_f_betaHat_beta(beta):
		log_p = log_f_betaHat_beta(beta, betaHat_obs, sigma, cutoff)
		minus_log_p = -log_p
		return minus_log_p
	#brent returns the argument that gets the local minimum (need to format argmax as argmin)
	mle = brent(minus_log_f_betaHat_beta)
	return mle

def generalizedExpectationF(betaTrue,sigma,cutoff):
		numerator = stdNormPdf(betaTrue/sigma-cutoff) - stdNormPdf(-betaTrue/sigma-cutoff)
		denominator = stdNormCdf(betaTrue/sigma-cutoff) + stdNormCdf(-betaTrue/sigma-cutoff)
		return betaTrue + (sigma*(numerator)/(denominator))

def findMean(betaHat_obs, sigma, cutoff):
	#betaHat_mean = beta : E(betaHat_obs; beta) = betaHat_obs ---- ":" == "such that."
	#brentq is a zero finding algorithm. brentq(function, a,b) where f(a) and f(b) must have different signs.
	def expectationF(betaTrue):
		return generalizedExpectationF(betaTrue, sigma, cutoff) - betaHat_obs
	a=0
	b=betaHat_obs*2
	betaHat_mean = brentq(expectationF,a,b) #find beta true s.t. expectation = betaHat_obs
	return betaHat_mean

def findMedian(betaHat_obs):
	#betaHat_mean = beta : E(betaHat_obs; beta) = betaHat_obs
	raise NotImplementedError
	betaHat_med = None
	return betaHat_med

def findMSE(betaHat_obs, sigmaHat, cutoff, using="MLE"):
	if using == "MLE":
		betaHat_corrected = findMLE(betaHat_obs, sigmaHat, cutoff)
	elif using == "Mean":
		betaHat_corrected = findMean(betaHat_obs, sigmaHat, cutoff)
	elif using == "Median":
		betaHat_corrected = findMedian(betaHat_obs)
	else:
		raise ValueError("Used algorithm is %s: this is an invalid option, use 'MLE' or 'Mean' or 'Median'." % using)
	obsWeightK = sigmaHat**2/(sigmaHat**2 + (betaHat_obs - betaHat_corrected)**2)
	betaHat_mse = obsWeightK*betaHat_obs + (1-obsWeightK)*betaHat_corrected
	return betaHat_mse

#load the data
if args.skipHeader:
	uncorrectedStats = pd.read_csv(args.summStats,sep="\t", header=None, names = colNames,skiprows=1)#,nrows=1000)
else:
	uncorrectedStats = pd.read_csv(args.summStats,sep="\t", header=None, names = colNames)

if args.forceCutoff:
	uncorrectedStats = uncorrectedStats[uncorrectedStats["p-value"] < gws_p]


####################################################### WC Correction #######################################################


uncorrectedStats["MarkerName"] = uncorrectedStats.apply(lambda row: "%s_%s_%s:%s" % (str(row["chr"]),str(row["pos"]),str(row["A0"]),str(row["A1"])),axis=1)
uncorrectedStats = uncorrectedStats.set_index("MarkerName")
print(uncorrectedStats.head())

uncorrectedStats["beta_Mean"] = uncorrectedStats.apply(lambda row: findMean(row["beta"],row["SE"],gws),axis=1)
uncorrectedStats["beta_MSE"] = uncorrectedStats.apply(lambda row: findMSE(row["beta"],row["SE"],gws),axis=1)
uncorrectedStats["beta_MLE"] = uncorrectedStats.apply(lambda row: findMLE(row["beta"],row["SE"],gws),axis=1)

uncorrectedStats["delta_original_Mean"] = uncorrectedStats.apply(lambda row: magCalc(row,"beta","beta_Mean"),axis=1)
uncorrectedStats["delta_original_MSE"] = uncorrectedStats.apply(lambda row: magCalc(row,"beta","beta_MSE"),axis=1)
uncorrectedStats["delta_original_MLE"] = uncorrectedStats.apply(lambda row: magCalc(row,"beta","beta_MLE"),axis=1)

uncorrectedStats.to_csv("%s_withCorrections.txt" % args.outPrefix,index=False,sep="\t")

if args.quiet == 0:
	print("Plotting.")
	boxesToDo = ["Mean","MLE","MSE"]

	plt.figure()
	plt.boxplot([[x[i] for x in uncorrectedStats[["delta_original_%s" % estimate for estimate in boxesToDo]].values] for i in list(range(0,len(boxesToDo)))],
		labels=["%s" % estimate for estimate in boxesToDo])
	plt.yscale('log')
	plt.xlabel("Beta estimation method")
	plt.ylabel(magYlabel)

	plt.title("Comparison to own dataset")

	plt.savefig("%s_internalBox.png" % (args.outPrefix))
	print("Done plotting for base data.")



####################################################### Comparison to Replication #######################################################

if args.summStatsMerged != None:
	uncorrectedStats = pd.read_csv(args.summStatsMerged,sep="\t")
	if args.forceCutoff:
		uncorrectedStats = uncorrectedStats[uncorrectedStats["p-value"] < 5e-8]
elif args.summStats2 != None:
	#load the comparison data
	print("Loading comparison data. This might take a while.")
	uncorrectedStats["replicateBeta"] = np.NaN
	uncorrectedStats["replicatePval"] = np.NaN
	matches = set()
	linesRead = 0
	foundMatch = 0
	betaIndex = otherColNames.index("beta")
	pvalIndex = otherColNames.index("p-value")
	usingMN=False
	if "MarkerName" in otherColNames:
		mnIndex = otherColNames.index("MarkerName")
		usingMN = True
	else:
		chrIndex=otherColNames.index("chr")
		posIndex=otherColNames.index("pos")
		a0Index=otherColNames.index("A0")
		a1Index=otherColNames.index("A1")
	with open(args.summStats2,'r') as f:
		f.readline()
		for line in f:
			linesRead += 1
			#print(linesRead)
			# if linesRead >10:# 35000000:
			# 	break
			if linesRead % 1000000 == 0:
				print(linesRead)
			splitline = line.split("\t")
			if usingMN:
				markerName = splitline[mnIndex]
			else:
				markerName = "%s_%s_%s:%s" % (splitline[chrIndex],splitline[posIndex],splitline[a0Index],splitline[a1Index])
			#print(markerName)
			beta = float(splitline[betaIndex].strip())
			pval = float(splitline[pvalIndex].strip())
			if markerName in uncorrectedStats.index:
				foundMatch += 1
				if foundMatch %1000 == 1:
					print(foundMatch)
				if markerName in matches:
					print ("%s already found..." % markerName)
					if pval < uncorrectedStats.loc[markerName,"replicatePval"]:
						uncorrectedStats.loc[markerName,"replicateBeta"] = beta
						uncorrectedStats.loc[markerName,"replicatePval"] = pval
						print("and its better!")
					else:
						print("and its not as good.")
				else:
					matches.add(markerName)
					#print(uncorrectedStats.loc[rsid])
					uncorrectedStats.loc[markerName,"replicateBeta"] = beta
					uncorrectedStats.loc[markerName,"replicatePval"] = pval
					#print(uncorrectedStats.loc[rsid]) #, 'Proximity'


	print("Of %s significant snps, %s have pvals in the duplicate dataset." % (uncorrectedStats.shape[0],uncorrectedStats[pd.isnull(uncorrectedStats["replicateBeta"])].shape[0]))




	uncorrectedStats["delta_original"] = abs(uncorrectedStats["replicateBeta"] - uncorrectedStats["beta"])
	uncorrectedStats["delta_Mean"] = abs(uncorrectedStats["replicateBeta"] - uncorrectedStats["beta_Mean"])
	uncorrectedStats["delta_MSE"] = abs(uncorrectedStats["replicateBeta"] - uncorrectedStats["beta_MSE"])
	uncorrectedStats["delta_MLE"] = abs(uncorrectedStats["replicateBeta"] - uncorrectedStats["beta_MLE"])

	uncorrectedStats.to_csv("%s_withReplicate.txt" % args.outPrefix,index=False,sep="\t")

else:
	exit()

print("plotting merged data.")
# from scipy.stats.stats import pearsonr
# pearsonr(justGWS["beta"],justGWS["replicateBeta"])

boxesToDo = ["original","Mean","MLE","MSE"]


#############################################
# Basic delta between estimate and replicate.
#############################################

# plt.figure()
# plt.yscale('log')
# plt.boxplot([[x[i] for x in uncorrectedStats[["delta_%s" % estimate for estimate in boxesToDo]].values] for i in list(range(0,len(boxesToDo)))],
# 	labels=["%s" % estimate for estimate in boxesToDo])

# plt.title("Comparison to replication dataset")
# plt.xlabel("Beta estimation method")
# plt.ylabel("| Repl_beta - estimate |")

# plt.savefig("%s_boxPlot.png" % (args.outPrefix))
# plt.close()



#########################################
# estimate vs replicate: delta vs pvalue.
#########################################

# boxesToDo = ["original","Mean","MLE","MSE"]
# colors = ["royalblue","tomato","forestgreen","mediumorchid"]

# plt.figure()
# plt.yscale('log')
# lines = []
# for name,color in zip(boxesToDo,colors):
# 	lines.append(plt.plot(-np.log10(uncorrectedStats["replicatePval"]),uncorrectedStats["delta_%s" % name], 'o',
# 		color=color,label=name,alpha=.2,ms=2)[0])

# leg=plt.legend(lines,boxesToDo)
# for l in leg.get_lines():
# 	l.set_alpha(1)
# 	l.set_marker('.')

# plt.title("Comparison to replication dataset")
# plt.xlabel("-log10 replication pval")
# plt.ylabel("| Repl_beta - estimate |")

# plt.savefig("%s_vsPvalRepl.png" % (args.outPrefix))
# plt.close()


#######################################################
# plots beta - estimate vs pvalue, in original dataset.
#######################################################

# boxesToDo = ["Mean","MLE","MSE"]
# colors = ["tomato","forestgreen","mediumorchid"]

# plt.figure()
# plt.yscale('log')
# lines = []
# for name,color in zip(boxesToDo,colors):
# 	lines.append(plt.plot(-np.log10(uncorrectedStats["p-value"]),uncorrectedStats["delta_original_%s" % name], 'o',
# 		color=color,label=name,alpha=.2,ms=2)[0])

# plt.axvline(x=-np.log10(gws_p),color='r')
# leg=plt.legend(lines,boxesToDo)
# for l in leg.get_lines():
# 	l.set_alpha(1)
# 	l.set_marker('.')

# plt.title("Delta vs Pval")
# plt.xlabel("-log10 pval")
# plt.ylabel(magYlabel)

# plt.savefig("%s_vsPval.png" % (args.outPrefix))
# plt.close()



#######################################################
# plots ratio of estimate to beta vs pvalue, in original dataset.
#######################################################
# boxesToDo = ["Mean","MLE","MSE"]
# colors = ["tomato","forestgreen","mediumorchid"]

# plt.figure()
# lines = []
# for name,color in zip(boxesToDo,colors):
# 	lines.append(plt.plot(-np.log10(uncorrectedStats["p-value"]),uncorrectedStats["beta_%s" % name]/uncorrectedStats["beta"], 'o',
# 		color=color,label=name,alpha=.2,ms=1)[0])

# plt.axvline(x=-np.log10(gws_p),color='r')
# leg=plt.legend(lines,boxesToDo)
# for l in leg.get_lines():
# 	l.set_alpha(1)
# 	l.set_marker('.')

# plt.title("Ratio vs Pval")
# plt.xlabel("-log10 pval")
# plt.ylabel("beta_estimate / beta")

# plt.savefig("%s_ratioVsPval.png" % (args.outPrefix))
# plt.close()



#### old test statistic
# uncorrectedStats["deltaMag_MLE"] = abs(uncorrectedStats["beta_MLE"])-abs(uncorrectedStats["replicateBeta"])
# uncorrectedStats["deltaMag_MSE"] = abs(uncorrectedStats["beta_MSE"])-abs(uncorrectedStats["replicateBeta"])
# uncorrectedStats["deltaMag_original"] = abs(uncorrectedStats["beta"])-abs(uncorrectedStats["replicateBeta"])
# uncorrectedStats["deltaMag_Mean"] = abs(uncorrectedStats["beta_Mean"])-abs(uncorrectedStats["replicateBeta"])

#### new test statistic
uncorrectedStats["deltaMag_MLE"] = uncorrectedStats.apply(lambda row: magCalcRepl(row,"beta_MLE"),axis=1)
uncorrectedStats["deltaMag_MSE"] = uncorrectedStats.apply(lambda row: magCalcRepl(row,"beta_MSE"),axis=1)
uncorrectedStats["deltaMag_original"] = uncorrectedStats.apply(lambda row: magCalcRepl(row,"beta"),axis=1)
uncorrectedStats["deltaMag_Mean"] = uncorrectedStats.apply(lambda row: magCalcRepl(row,"beta_Mean"),axis=1)
magYlabel = "abs(Estimate) - signed(Repl_beta)"



boxesToDo = ["original","Mean","MLE","MSE"]



#######################################################
# test statistic box plot, and zoomed version.
#######################################################
fig = plt.figure(figsize=(8,4))
ax = plt.subplot()
ax.boxplot([[x[i] for x in uncorrectedStats[["deltaMag_%s" % estimate for estimate in boxesToDo]].values] for i in list(range(0,len(boxesToDo)))],
	labels=[estimate for estimate in boxesToDo])
ax.axhline(y=0,color='r')
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.title("Comparison to replication dataset")
plt.xlabel("Beta estimation method")
plt.ylabel(magYlabel)

plt.savefig("%s_mag_boxPlot.png" % (args.outPrefix))
plt.close()


fig = plt.figure(figsize=(8,4))
ax = plt.subplot()
ax.boxplot([[x[i] for x in uncorrectedStats[["deltaMag_%s" % estimate for estimate in boxesToDo]].values] for i in list(range(0,len(boxesToDo)))],
	labels=[estimate for estimate in boxesToDo])
ax.axhline(y=0,color='r')
ax.set_ylim(-.025, .025)  # outliers only
ax.spines['bottom'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.title("Comparison to replication dataset")
plt.xlabel("Beta estimation method")
plt.ylabel(magYlabel)

plt.savefig("%s_mag_boxPlotZoom.png" % (args.outPrefix))
plt.close()


#######################################################
# test statistic box plot, and zoomed version; stratified by p-value.
#######################################################
strata = [1] + [gws_p/x for x in [2,5,10,20,50,100,500,1000,10000]] + [0]
datasets = [uncorrectedStats[(uncorrectedStats["p-value"] < strata[i]) & (uncorrectedStats["p-value"] > strata[i+1])] for i in range(len(strata)-1)]

print("Stratified N(snps) for %s" % args.outPrefix)
for i,j in zip(strata[1:],datasets):
	print(i)
	print(j.shape)


f,axarr = plt.subplots(len(boxesToDo),figsize=(12,6), sharex=True)
for index,estimate in enumerate(boxesToDo):
	print(index)
	axarr[index].axhline(y=0,color='r')
	axarr[index].boxplot([[dataset[["deltaMag_%s" % estimate]].values] for dataset in datasets],
		labels=['{:0.1e}'.format(x) for x in strata[1:]])
	# axarr[index].boxplot(datasets[0][["deltaMag_%s" % estimate]].values)
	axarr[index].set_ylim(-.025, .025)  # outliers only
	axarr[index].spines['bottom'].set_visible(False)
	axarr[index].spines['top'].set_visible(False)
	axarr[index].set_title(estimate)

plt.suptitle("Comparison to replication dataset")
plt.xlabel("Min Pval")
plt.ylabel(magYlabel)

plt.savefig("%s_mag_boxPlotZoom_stratified.png" % (args.outPrefix))
plt.close()






uncorrectedStats["-log10_p"] = -np.log10(uncorrectedStats["p-value"])
uncorrectedStats["-log10_p_Decile"] = pd.qcut(uncorrectedStats["-log10_p"],10)
uncorrectedStats["SEDecile"] = pd.qcut(uncorrectedStats["SE"],10)
uncorrectedStats["-log10_p_Quantile"] = pd.qcut(uncorrectedStats["-log10_p"],5)
uncorrectedStats["SEQuantile"] = pd.qcut(uncorrectedStats["SE"],5)



#######################################################
# test statistic box plot, and zoomed version; stratified by p-value AND Standard Error.
#######################################################
import seaborn as sns
f,axarr = plt.subplots(len(boxesToDo),figsize=(16,10), sharex=True)
for index,estimate in enumerate(boxesToDo):
	print(index)
	print(estimate)
	sns.boxplot(x='-log10_p_Quantile', y='deltaMag_%s' % estimate, hue='SEQuantile', data=uncorrectedStats,ax=axarr[index])
	axarr[index].set_ylim(-.02, .02)
	axarr[index].axhline(y=0,color='r')
	if index==0:
		axarr[index].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="SE_Quantile")
	else:
		axarr[index].get_legend().set_visible(False)

#plt.show()
plt.suptitle("%s, stratified by SE and pval." % magYlabel)
#plt.tight_layout()
#plt.xlabel("-log10(p) strata")
plt.savefig("%s_mag_boxPlotZoom_stratified2.png" % (args.outPrefix),bbox_inches="tight")
plt.close()



print("Done plotting.")