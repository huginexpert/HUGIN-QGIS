# -*- coding: utf-8 -*-

"""
/***************************************************************************
 HuginQGIS
                                 A QGIS plugin
 Perform spatial probabilistic analysis using HUGIN Bayesian Networks
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                              -------------------
        begin                : 2019-10-10
        copyright            : (C) 2019 by Hugin Expert A/S
        email                : qgis@hugin.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Hugin Expert A/S'
__date__ = '2019-10-10'
__copyright__ = '(C) 2019 by Hugin Expert A/S'

# This will get replaced with a git SHA1 when you do a git archive

__revision__ = '$Format:%H$'

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingException,
                       QgsProcessingParameterMultipleLayers,
                       QgsProcessingParameterFile,
                       QgsProcessingParameterRasterDestination)

import configparser
import numpy as np
import os

from osgeo import gdal
from gdalconst import *
import osr

import tempfile


def loadConfig(path):
    config = configparser.ConfigParser()
    # preserve case in ini-file property names
    config.optionxform = str
    config.read(path)
    return config

def loadRaster(path):
    ds = gdal.Open(path, GA_ReadOnly)
    if ds is None:
        raise QgsProcessingException("Cannot open raster '{}'.".format(path))
    return ds

def get_node_intervals(node):
    intervals = []
    for i in range(0, node.get_number_of_states() + 1):
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
        for k in range(1, stateCount):
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
            for k in range(1, stateCount):
                if value == intervals[k-1] or value < intervals[k]:
                    return k-1
            return stateCount - 1
        return func
    elif not lowgtinf and highltinf:
        def func(value):
            if value > high:
                return -1
            for k in range(1, stateCount):
                if value == intervals[k-1] or value < intervals[k]:
                    return k-1
            return stateCount - 1
        return func
    elif lowgtinf and highltinf:
        def func(value):
            if value < low or value > high:
                return -1
            for k in range(1, stateCount):
                if value == intervals[k-1] or value < intervals[k]:
                    return k-1
            return stateCount - 1
        return func
    else:
        def func(value):
            for k in range(1, stateCount):
                if value == intervals[k-1] or value < intervals[k]:
                    return k-1
            return stateCount - 1
        return func


class OutFunc(object):
    @staticmethod
    def MAX(node, stateCount):
        def f():
            maxIndex = 0
            maxValue = 0
            for i in range(0, stateCount):
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
            for i in range(0, stateCount):
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
        raise QgsProcessingException("sampling not implmented")
        count = node.get_number_of_states()
        #produce the distribution used in SAMPLE2
        def f():
            return [node.get_belief(i) for i in range(0, count)]
        return f

    @staticmethod
    def POST_SAMPLE(node):
        raise QgsProcessingException("sampling not implmented")
        count = node.get_number_of_states()
        intervals = get_node_intervals(node)
        def f(distribution):
            #TBD sample from distribution
            sampledStateIndex = random.random()
            return sampledStateIndex
        return f


def assert_raster_compatible(r1, name1, r2, name2):
    if r1.GetGeoTransform() != r2.GetGeoTransform():
        raise QgsProcessingException("Raster image files '{}' and '{}': incompatible transform.".format(name1, name2))
    if not osr.SpatialReference(r1.GetProjection()).IsSame(osr.SpatialReference(r2.GetProjection())):
        raise QgsProcessingException("Raster image files '{}' and '{}': incompatible projection.".format(name1, name2))
    if r1.RasterXSize != r2.RasterXSize:
        raise QgsProcessingException("Raster image files '{}' and '{}': incompatible RasterXSize {} != {}.".format(name1, name2, r1.RasterXSize, r2.RasterXSize))
    if r1.RasterYSize != r2.RasterYSize:
        raise QgsProcessingException("Raster image files '{}' and '{}': incompatible RasterYSize {} != {}.".format(name1, name2, r1.RasterYSize, r2.RasterYSize))

NoDataValue = -9999

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
    for i in range(0, outputDS.RasterCount):
        outputDS.GetRasterBand(i+1).SetNoDataValue(NoDataValue)
    outputDS.SetGeoTransform(referenceTrans)
    outputDS.SetProjection(referenceProj)
    res = gdal.ReprojectImage(inputDS, outputDS, inputProj, referenceProj, GRA_Bilinear)
    if res != 0:
        raise QgsProcessingException("GDAL ReprojectImage failed with {} on '{}'".format(res, outputFile))
    # flush changes to disk
    del outputDS
    return loadRaster(outputFile)

def applyPreFunctions(preFunctionList):
    for f in preFunctionList:
        yield(f())

def applyPostFunctions(postFunctionList, preFunctionResultList):
    for f, r in zip(postFunctionList, preFunctionResultList):
        yield r if f is None else f(r)

def create_cases_processor(domain, nodes, oPostFun, oPreFun, errorResult, CachedError, mustSPU):
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


def get_n_tuple(n, value):
    return tuple([value for i in range(0, n)])



class HuginQGISAlgorithm(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer and
    creates a new identical one.

    It is meant to be used as an example of how to create your own
    algorithms and explain methods and variables used to do it. An
    algorithm like this will be available in all elements, and there
    is no need for additional work.

    All Processing algorithms should extend the QgsProcessingAlgorithm
    class.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.

    INPUT_RASTERS = 'INPUT_RASTERS'
    HUGIN_NET_FILE = 'HUGIN_NET_FILE'
    HUGIN_CONFIGURATION_FILE = 'HUGIN_CONFIGURATION_FILE'
    OUTPUT_RASTER = 'OUTPUT_RASTER'

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """

        # The algorithm works on multiple raster layers.
        self.addParameter(
            QgsProcessingParameterMultipleLayers(
                self.INPUT_RASTERS,
                self.tr('Input rasters'),
                QgsProcessing.TypeRaster
            )
        )

        # Hugin NET file
        self.addParameter(
            QgsProcessingParameterFile(
                self.HUGIN_NET_FILE,
                self.tr('Hugin NET file')
            )
        )

        # Hugin configuration file
        self.addParameter(
            QgsProcessingParameterFile(
                self.HUGIN_CONFIGURATION_FILE,
                self.tr('Hugin configuration file')
            )
        )

        # 
        self.addParameter(
            QgsProcessingParameterRasterDestination(
                self.OUTPUT_RASTER,
                self.tr('Output raster')
            )
        )


    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """

        input_rasters = self.parameterAsLayerList (parameters, self.INPUT_RASTERS, context)
        configuration_file = self.parameterAsString (parameters, self.HUGIN_CONFIGURATION_FILE, context)
        HUGIN_Net_File = self.parameterAsString (parameters, self.HUGIN_NET_FILE, context)
        Output_Raster = self.parameterAsOutputLayer (parameters, self.OUTPUT_RASTER, context)

        domain = None
        classes = []
        discretizers = []

        try:
            feedback.setProgress(0)
            feedback.setProgressText("Parsing configuration file '{}'".format(configuration_file))
            config = loadConfig (configuration_file)

            #feedback.setProgressText("Asserting raster configuration file".format(configuration_file))
            if config.get("hugin", "type") != "raster":
                raise QgsProcessingException("not raster config file")

            pyhuginName = config.get("hugin", "pyhugin")
    
            feedback.setProgressText("Importing pyhugin module '{}'".format(pyhuginName))
            exec ("from {} import *".format(pyhuginName), globals())

            feedback.setProgressText("Loading Hugin NET file '{}'".format(HUGIN_Net_File))
            domain = Domain.parse_domain(HUGIN_Net_File)

            # Get the input Node objects
            feedback.pushInfo ("Input nodes:")
            nodes = []
            for nodeName, strBandIndex in config.items("input"):
                feedback.pushInfo("- {}".format(nodeName))
                n = domain.get_node_by_name(nodeName)
                # assert node exists
                if n is None:
                    raise QgsProcessingException("Cannot find node '{}'.".format(nodeName))
                # assert node is of type interval
                if n.get_subtype() is not SUBTYPE.INTERVAL:
                    raise QgsProcessingException("Wrong node type for '{}', discrete interval chance node required.".format(nodeName))
                nodes.append(n)
                intervals = get_node_intervals(n)
                feedback.pushInfo("  {}".format(intervals))
                classes.append(intervals)
                df = np.vectorize(get_discretizer2(intervals), otypes = [np.int16], cache = False)
                discretizers.append(df)

            nodeCount = len (nodes)

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

            feedback.pushInfo ("Output:")
            try:
                configItems = sorted(config.items("output"), key = lambda ci: int(ci[0]))
            except ValueError as e:
                raise QgsProcessingException("Output band index error, check config file. Indices must be integers: '{}'".format(e))

            index = 1
            for strIndex, strSpec in configItems:
                feedback.pushInfo(" #{}".format(index))
                if "{}".format(index) != "{}".format(strIndex):
                    raise QgsProcessingException("Output band index error, check config file. Expected '{}', but got '{}'".format(index, strIndex))
                arg = strSpec.split()
                func = arg.pop(0)
                if func == "MAX":
                    try:
                        name = arg.pop(0)
                    except Exception:
                        raise QgsProcessingException("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
                    n = domain.get_node_by_name(name)
                    if n is None:
                        raise QgsProcessingException("Cannot find node '{}'".format(name))

                    oPreFun.append(OutFunc.MAX(n, n.get_number_of_states()))
                    oPostFun.append(None)
                    feedback.pushInfo("  {} {}".format(func, name))
                elif func == "PMAX":
                    try:
                        name = arg.pop(0)
                    except Exception:
                        raise QgsProcessingException("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
                    n = domain.get_node_by_name(name)
                    if n is None:
                        raise QgsProcessingException("Cannot find node '{}'".format(name))
                    oPreFun.append(OutFunc.PMAX(n, n.get_number_of_states()))
                    oPostFun.append(None)
                    feedback.pushInfo("  {} {}".format(func, name))
                elif func == "MEU":
                    oPreFun.append(OutFunc.MEU(domain))
                    oPostFun.append(None)
                    feedback.pushInfo("  {}".format(func))
                elif func == "EU":
                    try:
                        name = arg.pop(0)
                        state = arg.pop(0)
                        state = int(state)
                    except Exception:
                        raise QgsProcessingException("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME INDEX'".format(func))
                    n = domain.get_node_by_name(name)
                    if n is None:
                        raise QgsProcessingException("Cannot find node '{}'".format(name))
                    oPreFun.append(OutFunc.EU(n, state))
                    oPostFun.append(None)
                    feedback.pushInfo("  {} {}=state {}".format(func, name, state))
                elif func == "AVG":
                    try:
                        name = arg.pop(0)
                    except Exception:
                        raise QgsProcessingException("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
                    n = domain.get_node_by_name(name)
                    if n is None:
                        raise QgsProcessingException("Cannot find node '{}'".format(name))
                    oPreFun.append(OutFunc.AVG(n))
                    oPostFun.append(None)
                    feedback.pushInfo("  {} {}".format(func, name))
                elif func == "VAR":
                    try:
                        name = arg.pop(0)
                    except Exception:
                        raise QgsProcessingException("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
                    n = domain.get_node_by_name(name)
                    if n is None:
                        raise QgsProcessingException("Cannot find node '{}'".format(name))
                    oPreFun.append(OutFunc.VAR(n))
                    oPostFun.append(None)
                    feedback.pushInfo("  {} {}".format(func, name))
                elif func == "QUANTILE":
                    try:
                        name = arg.pop(0)
                        probability = float(arg.pop(0))
                    except Exception:
                        raise QgsProcessingException("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME PROBABILITY'".format(func))
                    n = domain.get_node_by_name(name)
                    if n is None:
                        raise QgsProcessingException("Cannot find node '{}'".format(name))
                    oPreFun.append(OutFunc.QUANTILE(n, probability))
                    oPostFun.append(None)
                    feedback.pushInfo("  {} {}".format(func, probability))
                elif func == "SAMPLE":
                    try:
                        name = arg.pop(0)
                    except ValueError:
                        raise QgsProcessingException("Invalid arguments for output function '{0}'. Expected: '{0} NODENAME'".format(func))
                    n = domain.get_node_by_name(name)
                    if n is None:
                        raise QgsProcessingException("Cannot find node '{}'".format(name))

                    oPreFun.append(OutFunc.PRE_SAMPLE(n))
                    oPostFun.append(OutFunc.POST_SAMPLE(n))
                    feedback.pushInfo ("{}".format(get_node_intervals(n)))
                    feedback.pushInfo ("{}".format(n.get_number_of_states()))
                    feedback.pushInfo ("  {} {}".format(func, name))
                else:
                    raise QgsProcessingException("Unknown output function '{}'".format(func))
                if len(arg) != 0:
                    raise QgsProcessingException("Too many arguments specified for output function '{}', check config file".format(func))
                if len(oPreFun) == 0:
                    raise QgsProcessingException("No output functions specified")
                index = index + 1

            if all(f is None for f in oPostFun):
                # No post functions needed: We skip the post-step entirely, saving 10-15% execution time
                oPostFun = None

            feedback.setProgressText ("Compiling domain")
            domain.compile()

            feedback.setProgressText ("Aligning data and wiring to input nodes")

            rasters = dict()
            baseRaster = None
            bands = []
            tempdir = tempfile.mkdtemp()

            raster_paths = [layer.source() for layer in input_rasters]
            for nodeName, rasterName in config.items("input"):
                hit = False
                nameBand = rasterName.split("|")
                if len(nameBand) != 2:
                    raise QgsProcessingException("Not a valid specification of a raster file and band index '{}'".format(rasterName))
                band_index = int(nameBand[1])
                feedback.pushInfo ("- {} : '{}' band #{}".format(nodeName, nameBand[0], band_index))
                for path in raster_paths:
                    raster_name = os.path.basename(path)
                    if raster_name.find(nameBand[0]) >= 0:
                        r = rasters.get(path)
                        if r is None:
                            r = loadRaster(path)
                            if baseRaster is None:
                                baseRaster = r
                                baseRasterName = raster_name
                            else:
                                try:
                                    # only align if neccessary
                                    assert_raster_compatible (baseRaster, baseRasterName, r, raster_name)
                                except:
                                    outputFile = os.path.join(tempdir, raster_name)
                                    #feedback.pushInfo ("Output file for alignRaster: {}".format(outputFile))
                                    r = alignRaster (baseRaster, r, outputFile)
                            rasters[path] = r
                        if (band_index > r.RasterCount) or (band_index < 1):
                            raise QgsProcessingException("Invalid band index {}: raster '{}' contains {} bands.".format(band_index, raster_name, r.RasterCount))
                        bands.append (r.GetRasterBand(band_index))
                        hit = True
                        break

                if not hit:
                    raise QgsProcessingException("Could not identify a raster file that matches substring '{}'.".format(nameBand[0]))

            # Perform a final check that working raster images are compatible.
            # TBD should be deleted later when we are absolutely certain align step cannot produce unaligned rasters
            for path, r in list(rasters.items()):
                assert_raster_compatible (baseRaster, baseRasterName, r, path)

            # Prepare the output raster
            driver = gdal.GetDriverByName("GTiff")
            dstRaster = None
            outputBandCount = len(oPreFun)
            errorResult = get_n_tuple(outputBandCount, NoDataValue)
            dstRaster = driver.Create(Output_Raster, baseRaster.RasterXSize, baseRaster.RasterYSize, outputBandCount, gdal.GDT_Float32)
            for i in range(0, outputBandCount):
                dstRaster.GetRasterBand(i+1).SetNoDataValue(NoDataValue)
            dstRaster.SetGeoTransform(baseRaster.GetGeoTransform())
            dstRaster.SetProjection(baseRaster.GetProjection())

            width = baseRaster.RasterXSize
            rows = baseRaster.RasterYSize
            feedback.pushInfo("Data points: {} rasters x {}x{} points = {} total".format(len(rasters), width, rows, len(rasters)*width*rows))


            feedback.setProgressText("Computing")

            CachedError = NoDataValue
            process_row = create_cases_processor (domain, nodes, oPostFun, oPreFun, errorResult, CachedError, mustSPU)

            # Compute the number of steps to display within the progress bar
            total = 100.0 / rows
            prevPct = 0

            for r in range(0, rows):
                # Stop the algorithm if cancel button has been clicked
                if feedback.isCanceled():
                    break

                evidence = []

                # Read a single line from every input raster and discretize into node states
                for b, disc in zip(bands, discretizers):
                    rowData = b.ReadAsArray(0, r, width, 1)[0]
                    indices = disc(rowData) #TBD: optimize this as this is over half the processing time!
                    del rowData
                    evidence.append(indices)

                results = process_row(list(zip(*evidence)))

                # Write result to raster
                for i, data in zip(range(0, outputBandCount), list(map(list, list(zip(*results))))):
                    dstRaster.GetRasterBand(i+1).WriteArray(np.array([data]), 0, r)

                # Update the progress bar
                newPct = int (r * total)
                if newPct > prevPct:
                    feedback.setProgress (newPct)
                    prevPct = newPct


            # Flush and close the output file
            #dstRaster = None
            del dstRaster
            
            feedback.setProgress (100)
            feedback.setProgressText("Done!")

        finally:
            # We clean up after we are done
            if domain is not None:
                domain.delete()


        # Return the results of the algorithm. Some algorithms may return
        # multiple feature sinks, calculated numeric statistics, etc.
        # These should all be included in the returned dictionary, with keys
        # matching the feature corresponding parameter or output names.
        return {self.OUTPUT_RASTER: Output_Raster}

    def name(self):
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return 'huginbeliefupdate'

    def displayName(self):
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        #return self.tr(self.name())
        return 'Hugin belief update'

    def group(self):
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        #return self.tr(self.groupId())
        #return self.tr ('User scripts')
        return 'Algorithms for raster layers'

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        #return 'scripts'
        return 'rasteralgorithms'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        return HuginQGISAlgorithm()