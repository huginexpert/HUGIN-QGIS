#	This file is part of HUGIN QGIS Plugin that performs spatial
#	probabilistic analysis using HUGIN Bayesian Networks
#	
#	Copyright (c) 2018 Hugin Expert A/S
#	
#	HUGIN QGIS Plugin is free software: you can redistribute it and/or
#	modify it under the terms of the GNU General Public License as
#	published by the Free Software Foundation, either version 3 of the
#	License, or (at your option) any later version.
#	
#	HUGIN QGIS Plugin is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#	General Public License for more details.
#	
#	You should have received a copy of the GNU General Public License
#	along with HUGIN QGIS Plugin.  If not, see
#	<https://www.gnu.org/licenses/>.

##Input_Rasters=multiple raster
##HUGIN_Net_File=file
##HUGIN_Configuration_File=file
##Output_Raster=output raster

import ConfigParser
from osgeo import gdal
import osr
import _gdal
from gdalconst import *
import numpy as np
import os
import tempfile
import shutil
import random
from timeit import default_timer as timer

def loadConfig(path):
	config = ConfigParser.ConfigParser()
	# preserve case in ini-file property names
	config.optionxform = str
	config.read(HUGIN_Configuration_File)
	return config

def loadRaster(path):
	ds = gdal.Open(path, GA_ReadOnly)
	if ds is None:
		raise Exception("Cannot open raster '{}'.".format(path))
	return ds

def get_node_intervals(node):
	intervals = []
	for i in xrange(0, node.get_number_of_states() + 1):
		intervals.append(node.get_state_value(i))
	return intervals

def get_discretizer(intervals):
	low = intervals[0]
	high = intervals[len(intervals)-1]
	lowgtinf = low > float("-inf")
	highltinf = high < float("inf")
	stateCount = len(intervals)-1

	def discretizer(value):
		if (value < low and lowgtinf) or (value > high and highltinf):
			return -1
		for k in xrange(1, stateCount):
			if value == intervals[k-1] or value < intervals[k]:
				return k-1
		return stateCount - 1
	return discretizer

def get_discretizer2(intervals):
	low = intervals[0]
	high = intervals[len(intervals)-1]
	lowgtinf = low > float("-inf")
	highltinf = high < float("inf")
	stateCount = len(intervals)-1
	if lowgtinf and not highltinf:
		def func(value):
			if value < low:
				return -1
			for k in xrange(1, stateCount):
				if value == intervals[k-1] or value < intervals[k]:
					return k-1
			return stateCount - 1
		return func
	elif not lowgtinf and highltinf:
		def func(value):
			if value > high:
				return -1
			for k in xrange(1, stateCount):
				if value == intervals[k-1] or value < intervals[k]:
					return k-1
			return stateCount - 1
		return func
	elif lowgtinf and highltinf:
		def func(value):
			if value < low or value > high:
				return -1
			for k in xrange(1, stateCount):
				if value == intervals[k-1] or value < intervals[k]:
					return k-1
			return stateCount - 1
		return func
	else:
		def func(value):
			for k in xrange(1, stateCount):
				if value == intervals[k-1] or value < intervals[k]:
					return k-1
			return stateCount - 1
		return func

def get_node_beliefs(node, stateCount):
	return tuple([node.get_belief(i) for i in xrange(0, stateCount)])

def get_n_tuple(n, value):
	return tuple([value for i in xrange(0, n)])

def assert_raster_compatible(r1, name1, r2, name2):
	if r1.GetGeoTransform() != r2.GetGeoTransform():
		raise Exception("Raster image files '{}' and '{}': incompatible transform.".format(name1, name2))
	if not osr.SpatialReference(r1.GetProjection()).IsSame(osr.SpatialReference(r2.GetProjection())):
		raise Exception("Raster image files '{}' and '{}': incompatible projection.".format(name1, name2))
	if r1.RasterXSize != r2.RasterXSize:
		raise Exception("Raster image files '{}' and '{}': incompatible RasterXSize {} != {}.".format(name1, name2, r1.RasterXSize, r2.RasterXSize))
	if r1.RasterYSize != r2.RasterYSize:
		raise Exception("Raster image files '{}' and '{}': incompatible RasterYSize {} != {}.".format(name1, name2, r1.RasterYSize, r2.RasterYSize))

def alignRaster(refDS, inputDS, outputFile):
	inputProj = inputDS.GetProjection()
	inputTrans = inputDS.GetGeoTransform()
	referenceProj = refDS.GetProjection()
	referenceTrans = refDS.GetGeoTransform()
	bandreference = refDS.GetRasterBand(1)    
	x = refDS.RasterXSize 
	y = refDS.RasterYSize
	driver= gdal.GetDriverByName('GTiff')
	outputDS = driver.Create(outputFile, x, y, inputDS.RasterCount, bandreference.DataType)
	for i in xrange(0, outputDS.RasterCount):
		outputDS.GetRasterBand(i+1).SetNoDataValue(NoDataValue)
	outputDS.SetGeoTransform(referenceTrans)
	outputDS.SetProjection(referenceProj)
	res = gdal.ReprojectImage(inputDS, outputDS, inputProj, referenceProj, GRA_Bilinear)
	if res != 0:
		raise Exception("GDAL ReprojectImage failed with {} on '{}'".format(res, outputFile))
	# flush changes to disk
	del outputDS
	return loadRaster(outputFile)

def updatePercentage(v):
	progress.setPercentage(v)

def applyPreFunctions(preFunctionList):
	for f in preFunctionList:
		yield(f())

def applyPostFunctions(postFunctionList, preFunctionResultList):
	for f, r in zip(postFunctionList, preFunctionResultList):
		yield r if f is None else f(r)


NoDataValue = -9999
CachedError = NoDataValue
domain = None
dstRaster = None
rasters = dict()
bands = []
classes = []
discretizers = []
baseRaster = None
firtRasterName = None
tempdir = tempfile.mkdtemp()


class OutFunc:
	@staticmethod
	def MAX(node, stateCount):
		def f():
			maxIndex = 0
			maxValue = 0
			for i in xrange(0, stateCount):
				b = node.get_belief(i)
				if b > 0.5: #stop iterating node after state is definitely found
					return i
				if b > maxValue:
					maxValue = b
					maxIndex = i
			return maxIndex
		return f

	@staticmethod
	def PMAX(node, stateCount):
		def f():
			maxIndex = 0
			maxValue = 0
			for i in xrange(0, stateCount):
				b = node.get_belief(i)
				if b > 0.5: #stop iterating node after state is definitely found
					return b
				if b > maxValue:
					maxValue = b
					maxIndex = i
			return maxValue
		return f

	@staticmethod
	def MEU(domain):
		def f():
			return domain.get_expected_utility()
		return f

	@staticmethod
	def EU(node, state):
		def f():
			return node.get_expected_utility(state)
		return f

	@staticmethod
	def AVG(node):
		def f():
			return node.get_mean()
		return f

	@staticmethod
	def VAR(node):
		def f():
			return node.get_variance()
		return f

	@staticmethod
	def QUANTILE(node, probability):
		def f():
			return node.get_quantile(probability)
		return f
	
	@staticmethod
	def PRE_SAMPLE(node):
		raise Exception("sampling not implmented")
		count = node.get_number_of_states()
		#produce the distribution used in SAMPLE2
		def f():
			return [node.get_belief(i) for i in xrange(0, count)]
		return f

	@staticmethod
	def POST_SAMPLE(node):
		raise Exception("sampling not implmented")
		count = node.get_number_of_states()
		intervals = get_node_intervals(node)
		def f(distribution):
			#TBD sample from distribution
			sampledStateIndex = random.random()
			return sampledStateIndex
		return f

	
npsum = np.sum
	
def collect_beliefs_for_sampling(oBeliefNodes):
	result = []
	for i in oBeliefNodes:
		r = []
		for n, stateCount in i:
			b = get_node_beliefs(n, stateCount)
			normalized = b / np.sum(b)
			r.append(normalized)
		result.append(r)
	return result
		

def create_cases_processor(domain, nodes, oPostFun, oPreFun, errorResult, CachedError):
	cache = dict()
	cache_get = cache.get
	def func(cases):
		results = []
		results_append = results.append
		for theCase in cases:
			preResult = cache_get(theCase)
			if preResult is CachedError: #skip re-computing error data points
				results_append(errorResult)
				continue
			elif preResult: #we have a result
				if oPostFun is None:
					#skip running post-functions and save 10-15% execution time
					results_append(preResult)
					continue
				else:
					results_append(tuple(applyPostFunctions(oPostFun, preResult)))
					continue
			else: #we dont have a result - compute it!
				if all(v != -1 for v in theCase):
					#enter evidence
					for node, state in zip(nodes, theCase):
						node.select_state(state)
					#compute results
					try:
						domain.propagate()
						if mustSPU:
							domin.update_policies()
						preResult = tuple(applyPreFunctions(oPreFun))
						cache[theCase] = preResult
						if oPostFun is None:
							#skip running post-functions and save 10-15% execution time
							results_append(preResult)
							continue
						else:
							results_append(tuple(applyPostFunctions(oPostFun, preResult)))
							continue
					except HuginException:
						cache[theCase] = CachedError
						results_append(errorResult)
						continue;
				else: #incomplete case - we treat it as an error for now
					cache[theCase] = CachedError
					results_append(errorResult)
					continue
		return results
	return func

try:
	progress.setPercentage(0)
	progress.setText("Parse config file '{}'".format(HUGIN_Configuration_File))
	config = loadConfig(HUGIN_Configuration_File)

	progress.setText("Assert vector config file".format(HUGIN_Configuration_File))
	if config.get("hugin", "type") != "raster":
		raise Exception("not raster config file")

	pyhuginName = config.get("hugin", "pyhugin")
	
	progress.setText("Import pyhugin module '{}'".format(pyhuginName))
	exec ("from {} import *".format(pyhuginName))


	progress.setText("Load HUGIN net file '{}'".format(HUGIN_Net_File))
	domain = Domain.parse_domain(HUGIN_Net_File)


	# Get the input Node objects
	progress.setInfo("Input nodes:")
	nodes = []
	nodeCount = 0
	for nodeName, strBandIndex in config.items("input"):
		progress.setInfo("- {}".format(nodeName))
		n = domain.get_node_by_name(nodeName)
		# assert node exists
		if n is None:
			raise Exception("Cannot find node '{}'.".format(nodeName))
		# assert node is of type interval
		if n.get_subtype() is not SUBTYPE.INTERVAL:
			raise Exception("Wrong node type for '{}', discrete interval chance node required.".format(nodeName))
		nodes.append(n)
		nodeCount = nodeCount + 1
		intervals = get_node_intervals(n)
		progress.setInfo("- {}".format(intervals))
		classes.append(intervals)
		
		df = np.vectorize(get_discretizer2(intervals), otypes = [np.int16], cache = False)
		discretizers.append(df)
		
	# check for single policy updates if domain has decisions
	mustSPU = False
	for node in domain.get_nodes():
		if node.get_category() == CATEGORY.DECISION:
			mustSPU = True
			break

	# pre-run functions are applied and result is cached
	oPreFun = []
	# post-run functions are applied to result of pre-run functions
	oPostFun = []

	progress.setInfo("Output:")
	try:
		configItems = sorted(config.items("output"), key = lambda ci: int(ci[0]))
	except ValueError as e:
		raise Exception("Output band index error, check config file. Indices must be integers: '{}'".format(e))
	index = 1
	for strIndex, strSpec in configItems:
		progress.setInfo(" #{}".format(index))
		if "{}".format(index) != "{}".format(strIndex):
			raise Exception("Output band index error, check config file. Expected '{}', but got '{}'".format(index, strIndex))
		arg = strSpec.split()
		func = arg.pop(0)
		if func == "MAX":
			try:
				name = arg.pop(0)
			except Exception:
				raise Exception("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
			n = domain.get_node_by_name(name)
			if n is None:
				raise Exception("Cannot find node '{}'".format(name))

			oPreFun.append(OutFunc.MAX(n, n.get_number_of_states()))
			oPostFun.append(None)
			progress.setInfo("  {} {}".format(func, name))
		elif func == "PMAX":
			try:
				name = arg.pop(0)
			except Exception:
				raise Exception("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
			n = domain.get_node_by_name(name)
			if n is None:
				raise Exception("Cannot find node '{}'".format(name))
			oPreFun.append(OutFunc.PMAX(n, n.get_number_of_states()))
			oPostFun.append(None)
			progress.setInfo("  {} {}".format(func, name))
		elif func == "MEU":
			oPreFun.append(OutFunc.MEU(domain))
			oPostFun.append(None)
			progress.setInfo("  {}".format(func))
		elif func == "EU":
			try:
				name = arg.pop(0)
				state = arg.pop(0)
				state = int(state)
			except Exception:
				raise Exception("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME INDEX'".format(func))
			n = domain.get_node_by_name(name)
			if n is None:
				raise Exception("Cannot find node '{}'".format(name))
			oPreFun.append(OutFunc.EU(n, state))
			oPostFun.append(None)
			progress.setInfo("  {} {}=state {}".format(func, name, state))
		elif func == "AVG":
			try:
				name = arg.pop(0)
			except Exception:
				raise Exception("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
			n = domain.get_node_by_name(name)
			if n is None:
				raise Exception("Cannot find node '{}'".format(name))
			oPreFun.append(OutFunc.AVG(n))
			oPostFun.append(None)
			progress.setInfo("  {} {}".format(func, name))
		elif func == "VAR":
			try:
				name = arg.pop(0)
			except Exception:
				raise Exception("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
			n = domain.get_node_by_name(name)
			if n is None:
				raise Exception("Cannot find node '{}'".format(name))
			oPreFun.append(OutFunc.VAR(n))
			oPostFun.append(None)
			progress.setInfo("  {} {}".format(func, name))
		elif func == "QUANTILE":
			try:
				name = arg.pop(0)
				probability = arg.pop(0)
				probability = float(probability)
			except Exception:
				raise Exception("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME PROBABILITY'".format(func))
			n = domain.get_node_by_name(name)
			if n is None:
				raise Exception("Cannot find node '{}'".format(name))
			oPreFun.append(OutFunc.QUANTILE(n, probability))
			oPostFun.append(None)
			progress.setInfo("  {} {}".format(func, probability))
		elif func == "SAMPLE":
			try:
				name = arg.pop(0)
			except ValueError:
				raise Exception("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
			n = domain.get_node_by_name(name)
			if n is None:
				raise Exception("Cannot find node '{}'".format(name))

			oPreFun.append(OutFunc.PRE_SAMPLE(n))
			oPostFun.append(OutFunc.POST_SAMPLE(n))
			print("{}".format(get_node_intervals(n)))
			print("{}".format(n.get_number_of_states()))
			progress.setInfo("  {} {}".format(func, name))
		else:
			raise Exception("Unknown output function '{}'".format(func))
		if len(arg) != 0:
			raise Exception("Too many arguments specified for output function '{}', check config file".format(func))
		if len(oPreFun) == 0:
			raise Exception("No output functions specified")
		index = index + 1
		
	if all(f is None for f in oPostFun):
		#no post functions needed we skip the post-step entirely saving 10-15% execution time
		oPostFun = None

	progress.setText("Compile domain")
	domain.compile()

	progress.setText("Align data and wire to input nodes")
	for nodeName, rasterName in config.items("input"):
		hit = False
		nameBand = rasterName.split("|")
		if len(nameBand) != 2:
			raise Exception("Not a valid specification of a raster file and band index '{}'".format(rasterName))
		bandIndex = int(nameBand[1])
		progress.setInfo("- {} : '{}' band# {}".format(nodeName, nameBand[0], bandIndex))
		for path in Input_Rasters.split(";"):
			if os.path.basename(path).find(nameBand[0]) >= 0:
				r = rasters.get(path)
				if r is None:
					r = loadRaster(path)
					if (bandIndex > r.RasterCount) or (bandIndex < 1):
						raise Exception("Invalid band index {} indicated in '{}', raster '{}' contains {} bands.".format(bandIndex, os.path.basename(path), r.RasterCount))
					if baseRaster is None:
						baseRaster = r
						baseRasterName = os.path.basename(path)
					else:
						try:
							#only align if neccessary
							assert_raster_compatible(baseRaster, baseRasterName, r, os.path.basename(path))
						except:
							r = alignRaster(baseRaster, r, os.path.join(tempdir, os.path.basename(path)))
					rasters[path] = r
				bands.append(r.GetRasterBand(bandIndex))
				hit = True
				break
		if not hit:
			raise Exception("Could not identify a raster file that matches substring '{}'.".format(nameBand[0]))

	# perform a final check that working raster images are compatible. TBD should be deleted later when we are absolutely certain align step cannot produce unaligned rasters
	for path, r in rasters.items():
		assert_raster_compatible(baseRaster, baseRasterName, r, path)

	# prepare the output raster
	driver = gdal.GetDriverByName("GTiff")
	dstRaster = None
	outputBandCount = len(oPreFun)
	errorResult = get_n_tuple(outputBandCount, NoDataValue)
	dstRaster = driver.Create(Output_Raster, baseRaster.RasterXSize, baseRaster.RasterYSize, outputBandCount, gdal.GDT_Float32)
	for i in xrange(0, outputBandCount):
		dstRaster.GetRasterBand(i+1).SetNoDataValue(NoDataValue)
	dstRaster.SetGeoTransform(baseRaster.GetGeoTransform())
	dstRaster.SetProjection(baseRaster.GetProjection())

	width = baseRaster.RasterXSize
	rows = baseRaster.RasterYSize
	progress.setInfo("Data points: {} rasters x {}x{} points = {} total".format(len(rasters), width, rows, len(rasters)*width*rows))


	progress.setText("Computing")

	process_row = create_cases_processor(domain, nodes, oPostFun, oPreFun, errorResult, CachedError)
	startTime = timer()


	for r in xrange(0, baseRaster.RasterYSize):
		if r % 100 == 0: #update percentage indicator at low rate no need to spam system
			updatePercentage((float(r)/float(rows)) * 100)
		evidence = []

		# read a single line from every input raster and discretize into node states
		for b, disc in zip(bands, discretizers):
			rowData = b.ReadAsArray(0, r, width, 1)[0]
			indices = disc(rowData) #TBD: optimize this as this is over half the processing time!
			del rowData
			evidence.append(indices)

		results = process_row(zip(*evidence))

		# write result to raster
		for i, data in zip(xrange(0, outputBandCount), map(list, zip(*results))):
			dstRaster.GetRasterBand(i+1).WriteArray(np.array([data]), 0, r)

	endTime = timer()
	print("Duration {}".format(endTime - startTime))

	dstRaster = None
	updatePercentage(100)
	progress.setText("Done!")

finally:
	# we clean up after were done
	if domain is not None:
		domain.delete()
	if baseRaster is not None:
		del baseRaster
	if rasters is not None:
		del rasters
	if bands is not None:
		del bands
	if tempdir is not None:
		shutil.rmtree(tempdir)
