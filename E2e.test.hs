import System.Random

import Layer
import Network
import Test

g = mkStdGen 1337

testSimpleLinearNetwork :: String
testSimpleLinearNetwork =
	let
		-- f(x) = 5x + 4
		testName = "simpleLinearNetwork"

		is = [[0], [1], [2]] :: [LayerInput]
		os = [[4], [9], [14]] :: [LayerInput]
		alphaValue = 0.1
		inputWidth = 1
		specs = [LinearLayerSpecification 1]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 1000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
		expectedWeights = [[5]]
		expectedBiases = [4]
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr
		++ "; " ++ checkAllDeepApproxEqual (testName ++ " (weights)") expectedWeights resultWeights
		++ "; " ++ checkAllApproxEqual (testName ++ " (biases)") expectedBiases resultBiases

testLinearNetworkNOutputs :: String
testLinearNetworkNOutputs =
	let
		-- f(x) = [5x - 4, -2x + 7]
		testName = "linearNetworkNOutputs"

		is = [[0], [1], [2]] :: [LayerInput]
		os = [[-4, 7], [1, 5], [6, 3]] :: [LayerInput]
		alphaValue = 0.1
		inputWidth = 1
		specs = [LinearLayerSpecification 2]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 1000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
		expectedWeights = [[5], [-2]]
		expectedBiases = [-4, 7]
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr
		++ "; " ++ checkAllDeepApproxEqual (testName ++ " (weights)") expectedWeights resultWeights
		++ "; " ++ checkAllApproxEqual (testName ++ " (biases)") expectedBiases resultBiases

testLinearNetworkNInputs :: String
testLinearNetworkNInputs =
	let
		-- f(x, y) = (5x - 2y) + 3
		testName = "linearNetworkNInputs"

		is = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]] :: [LayerInput]
		os = [[3], [1], [-1], [8], [6], [4], [13], [11], [9]] :: [LayerInput]
		alphaValue = 0.04
		inputWidth = 2
		specs = [LinearLayerSpecification 1]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 1000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
		expectedWeights = [[5, -2]]
		expectedBiases = [3]
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr
		++ "; " ++ checkAllDeepApproxEqual (testName ++ " (weights)") expectedWeights resultWeights
		++ "; " ++ checkAllApproxEqual (testName ++ " (biases)") expectedBiases resultBiases

testLinearNetworkNInputsNOutputs :: String
testLinearNetworkNInputsNOutputs =
	let
		-- f(x, y) = [(5x - 2y) + 3, (-4x + 7y) -8]
		testName = "linearNetworkNInputsNOutputs"

		is = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]] :: [LayerInput]
		os = [[3, -8], [1, -1], [-1, 6], [8,-12], [6,-5], [4,2], [13,-16], [11,-9], [9,-2]] :: [LayerInput]
		alphaValue = 0.04
		inputWidth = 2
		specs = [LinearLayerSpecification 2]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 1000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
		expectedWeights = [[5, -2], [-4, 7]]
		expectedBiases = [3, -8]
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr
		++ "; " ++ checkAllDeepApproxEqual (testName ++ " (weights)") expectedWeights resultWeights
		++ "; " ++ checkAllApproxEqual (testName ++ " (biases)") expectedBiases resultBiases

testLinearNetworkNLayers :: String
testLinearNetworkNLayers =
	let
		-- f(x) = 5x + 4
		-- It is not necessary to have more than one layer but this test checks correct
		-- backpropagation of errors in linear layers
		testName = "linearNetworkNLayers"

		is = [[0], [1], [2]] :: [LayerInput]
		os = [[4], [9], [14]] :: [LayerInput]
		alphaValue = 0.01
		inputWidth = 1
		specs = [LinearLayerSpecification 1, LinearLayerSpecification 1]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 1000
		trainedLayer = head $ layers trainedNetwork

		expectedErr = 0
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr

testSimpleReluNetwork :: String
testSimpleReluNetwork =
	let
		-- f(x) = max(0, 5x + 4)
		testName = "simpleReluNetwork"

		is = [[-3], [-2], [-1], [0], [1], [2], [3]] :: [LayerInput]
		os = [[0], [0], [0], [0], [1], [2], [3]] :: [LayerInput]
		-- g=1337 results in a negative initial weight and a negative initial bias.
		-- This means the network cannot learn the pattern.
		gPositive = mkStdGen 1339
		alphaValue = 0.04
		inputWidth = 1
		specs = [LinearLayerSpecification 1, NonLinearLayerSpecification "relu"]
		network = createNetwork gPositive alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 4000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
		expectedWeights = [[1]]
		expectedBiases = [0]
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr
		++ "; " ++ checkAllDeepApproxEqual (testName ++ " (weights)") expectedWeights resultWeights
		++ "; " ++ checkAllApproxEqual (testName ++ " (biases)") expectedBiases resultBiases

testSimpleSigmoidNetwork :: String
testSimpleSigmoidNetwork =
	let
		-- f(x) = max(0, 5x + 4)
		testName = "simpleSigmoidNetwork"

		is = [[-3], [-2], [-1], [0], [1], [2], [3]] :: [LayerInput]
		os = [[0], [0], [0], [0.5], [1], [1], [1]] :: [LayerInput]
		alphaValue = 0.4
		inputWidth = 1
		specs = [LinearLayerSpecification 1, NonLinearLayerSpecification "sigmoid"]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 2000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
		-- Any positive weight will approximate the desired function, so we can't check the value
		expectedBiases = [0]
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr
		++ "; " ++ checkAllApproxEqual (testName ++ " (biases)") expectedBiases resultBiases

testLogicalAnd :: String
testLogicalAnd =
	let
		-- f(x, y) = x AND y
		testName = "logicalAnd"

		is = [[0,0], [0,1], [1,0], [1,1]] :: [LayerInput]
		os = [[0], [0], [0], [1]] :: [LayerInput]
		alphaValue = 0.1
		inputWidth = 2
		specs =
			[ LinearLayerSpecification 4
			, NonLinearLayerSpecification "sigmoid"
			, LinearLayerSpecification 1
			, NonLinearLayerSpecification "sigmoid"
			]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 10000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr

testLogicalThreeWayXor :: String
testLogicalThreeWayXor =
	let
		-- f(x, y, z) = x XOR y XOR z
		testName = "logicalThreeWayXor"

		is = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]] :: [LayerInput]
		os = [[0], [1], [1], [0], [1], [0], [0], [1]] :: [LayerInput]
		alphaValue = 0.1
		inputWidth = 3
		specs =
			[ LinearLayerSpecification 4
			, NonLinearLayerSpecification "sigmoid"
			, LinearLayerSpecification 1
			, NonLinearLayerSpecification "sigmoid"
			]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 10000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr

testLogicalThreeWayXorAndOr :: String
testLogicalThreeWayXorAndOr =
	let
		-- f(x, y, z) = [x XOR y XOR z, x AND y AND z, x OR y OR z]
		testName = "logicalThreeWayXorAndOr"

		is = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]] :: [LayerInput]
		os = [[0, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1], [0, 0, 1], [1, 1, 1]] :: [LayerInput]
		alphaValue = 0.1
		inputWidth = 3
		specs =
			[ LinearLayerSpecification 4
			, NonLinearLayerSpecification "sigmoid"
			, LinearLayerSpecification 3
			, NonLinearLayerSpecification "sigmoid"
			]
		network = createNetwork g alphaValue inputWidth specs
		(trainedNetwork, resultErr) = run is os network 10000
		trainedLayer = head $ layers trainedNetwork
		resultWeights = weights trainedLayer
		resultBiases = biases trainedLayer

		expectedErr = 0
	in checkApproxEqual (testName ++ " (err)") expectedErr resultErr


testE2eNetwork =
	[ testSimpleLinearNetwork
	, testLinearNetworkNOutputs
	, testLinearNetworkNInputs
	, testLinearNetworkNInputsNOutputs
	, testLinearNetworkNLayers
	, testSimpleReluNetwork
	, testSimpleSigmoidNetwork
	, testLogicalAnd
	, testLogicalThreeWayXor
	, testLogicalThreeWayXorAndOr
	]

main = do
	mapM putStrLn testE2eNetwork
