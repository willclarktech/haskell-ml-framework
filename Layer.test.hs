import Layer
import Math

approxEqual :: Float -> Float -> Bool
approxEqual m n = abs (m - n) < 0.000001

allApproxEqual :: [Float] -> [Float] -> Bool
allApproxEqual ms ns
	| length ms == length ns = all (== True) $ zipWith approxEqual ms ns
	| otherwise = False

allDeepApproxEqual :: [[Float]] -> [[Float]] -> Bool
allDeepApproxEqual ms ns
	| length ms == length ns = all (== True) $ zipWith allApproxEqual ms ns
	| otherwise = False

maybeAllDeepApproxEqual :: Maybe [[Float]] -> Maybe [[Float]] -> Bool
maybeAllDeepApproxEqual Nothing Nothing = True
maybeAllDeepApproxEqual (Just maybeDeepFloats1) (Just maybeDeepFloats2) = allDeepApproxEqual maybeDeepFloats1 maybeDeepFloats2
maybeAllDeepApproxEqual expected result = False

layerApproxEqual :: Layer -> Layer -> Bool
layerApproxEqual (LinearLayer activations1 inputs1 weights1 biases1) (LinearLayer activations2 inputs2 weights2 biases2) =
	maybeAllDeepApproxEqual activations1 activations2
	&& maybeAllDeepApproxEqual inputs1 inputs2
	&& allDeepApproxEqual weights1 weights2
	&& allApproxEqual biases1 biases2
layerApproxEqual (NonLinearLayer activations1 inputs1 fn1) (NonLinearLayer activations2 inputs2 fn2) =
	maybeAllDeepApproxEqual activations1 activations2
	&& maybeAllDeepApproxEqual inputs1 inputs2
	&& fn1 == fn2
layerApproxEqual _ _ = error "Not implemented"

layersApproxEqual :: [Layer] -> [Layer] -> Bool
layersApproxEqual layer1 layer2
	| length layer1 == length layer2 = all (== True) $ zipWith layerApproxEqual layer1 layer2
	| otherwise = False

createError :: (Show a) => String -> a -> a -> b
createError name expected result = error $ "Test " ++ name ++ " failed: expected " ++ show expected ++ " got " ++ show result

check :: (Show a) => (a -> a -> Bool) -> String -> a -> a -> String
check condition name expected result = if condition expected result
	then "Passed: " ++ name
	else createError name expected result

checkApproxEqual :: String -> Float -> Float -> String
checkApproxEqual = check approxEqual

checkAllApproxEqual :: String -> [Float] -> [Float] -> String
checkAllApproxEqual = check allApproxEqual

checkAllDeepApproxEqual :: String -> [[Float]] -> [[Float]] -> String
checkAllDeepApproxEqual = check allDeepApproxEqual

checkLayerApproxEqual :: String -> Layer -> Layer -> String
checkLayerApproxEqual = check layerApproxEqual

checkLayersApproxEqual :: String -> [Layer] -> [Layer] -> String
checkLayersApproxEqual = check layersApproxEqual

-- Tests

testActivateLayerLinear :: String
testActivateLayerLinear =
	let
		testName = "activateLayer - linear"

		inputs = [[0.8, 0.5, -0.2]]
		weights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		biases = [-0.1, -0.7]
		layer = LinearLayer Nothing Nothing weights biases
		result = activateLayer inputs layer

		expectedActivations = Just [[0.73, -0.41]]
		expectedInputs = Just inputs
		expected = LinearLayer expectedActivations expectedInputs weights biases
	in checkLayerApproxEqual testName expected result

testActivateLayerNonLinear :: String
testActivateLayerNonLinear =
	let
		testName = "activateLayer - non-linear"

		inputs = [[0.73, -0.41]]
		layer = NonLinearLayer Nothing Nothing relu
		result = activateLayer inputs layer

		expectedActivations = Just [[0.73, 0]]
		expectedInputs = Just inputs
		expected = NonLinearLayer expectedActivations expectedInputs relu
	in checkLayerApproxEqual testName expected result

testUpdateWeight :: String
testUpdateWeight =
	let
		testName = "updateWeight"

		alpha = 0.1
		err = (-0.54)
		activation = 0.8
		weight = 0.9
		result = updateWeight alpha err activation weight

		expected = 0.9432
	in checkApproxEqual testName expected result

testUpdatePartialWeights :: String
testUpdatePartialWeights =
	let
		testName = "updatePartialWeights"

		alpha = 0.1
		err = (-0.54)
		input = [0.8, 0.5, -0.2]
		weights = [0.9, -0.1, -0.8]
		result = updatePartialWeights alpha input err weights

		expected = [0.9432, -0.073, -0.8108]
	in checkAllApproxEqual testName expected result

testUpdateBias :: String
testUpdateBias =
	let
		testName = "updateBias"

		alpha = 0.1
		err = (-0.54)
		bias = (-0.1)
		result = updateBias alpha err bias

		expected = (-0.046)
	in checkApproxEqual testName expected result

testCalculateMeanInput :: String
testCalculateMeanInput =
	let
		testName = "calculateMeanInput"

		inputs = [[0.0, 0.0, 0.0], [1.0, 1.5, 1.0], [2.0, 0.0, -3.4]]
		result = calculateMeanInput inputs

		expected = [1.0, 0.5, -0.8]
	in checkAllApproxEqual testName expected result

testCalculateNextErrors :: String
testCalculateNextErrors =
	let
		testName = "calculateNextErrors"

		weights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		errors = [-0.54, 1.08]
		result = calculateNextErrors weights errors

		expected = [-0.27, 0.594, 1.08]
	in checkAllApproxEqual testName expected result

testUpdateLayerLinear :: String
testUpdateLayerLinear =
	let
		testName = "updateLayer - linear"

		alpha = 0.1
		activations = Just [[0.73, 0.59]]
		inputs = Just [[0.8, 0.5, -0.2]]
		weights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		biases = [-0.1, 0.2]
		layer = LinearLayer activations inputs weights biases
		errors = [-0.54, 1.08]
		(resultLayer, resultErrors) = updateLayer alpha layer errors

		expectedWeights = [[0.9432, -0.073, -0.8108], [0.1136, 0.446, 0.6216]]
		expectedBiases = [-0.046, 0.092]
		expectedLayer = LinearLayer Nothing Nothing expectedWeights expectedBiases
		expectedErrors = [-0.27, 0.594, 1.08]
	in
		checkLayerApproxEqual (testName ++ " (layer)") expectedLayer resultLayer
		++ "; "
		++ checkAllApproxEqual (testName ++ " (errors)") expectedErrors resultErrors

testUpdateLayerNonLinear :: String
testUpdateLayerNonLinear =
	let
		testName = "updateLayer - non-linear"

		alpha = 0.1
		activations = Just [[0.6748052, 0.64336514]]
		inputs = Just [[0.73, 0.59]]
		layer = NonLinearLayer activations inputs sigmoid
		errors = [-0.65038955, 1.2867303]
		(resultLayer, resultErrors) = updateLayer alpha layer errors

		expectedLayer = NonLinearLayer Nothing Nothing sigmoid
		expectedErrors = [-0.12819178, 0.31126007]
	in
		checkLayerApproxEqual testName expectedLayer resultLayer
		++ "; "
		++ checkAllApproxEqual testName expectedErrors resultErrors

testUpdateNextLayer :: String
testUpdateNextLayer =
	let
		testName = "updateNextLayer"

		alpha = 0.1
		activations = Just [[0.73, -0.41]]
		inputs = Just [[0.8, 0.5, -0.2]]
		weights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		biases = [-0.1, -0.7]
		layer = LinearLayer activations inputs weights biases
		errors = [1.46, 0]
		previousLayers = [NonLinearLayer Nothing Nothing relu]
		(resultErrors, resultLayers) = updateNextLayer alpha layer (errors, previousLayers)

		expectedErrors = [1.314, -0.146, -1.168]
		expectedLayers = [LinearLayer Nothing Nothing [[0.7832, -0.173, -0.7708], [0.2, 0.5, 0.6]] [-0.246, -0.7], NonLinearLayer Nothing Nothing relu]
	in
		checkAllApproxEqual (testName ++ " (errors)") expectedErrors resultErrors
		++ "; "
		++ checkLayersApproxEqual (testName ++ " (layers)") expectedLayers resultLayers

testModuleLayer :: [String]
testModuleLayer =
	[ testActivateLayerLinear
	, testActivateLayerNonLinear
	, testUpdateWeight
	, testUpdatePartialWeights
	, testUpdateBias
	, testCalculateMeanInput
	, testCalculateNextErrors
	, testUpdateLayerLinear
	, testUpdateLayerNonLinear
	, testUpdateNextLayer
	]

main = do
	mapM putStrLn testModuleLayer
