import Layer
import Math
import Network
import Test

testGetOutputWidthEmpty :: String
testGetOutputWidthEmpty =
	let
		testName = "getOutputWidth - empty"

		inputWidth = 4
		layers = []
		result = getOutputWidth inputWidth layers

		expected = 4
	in checkEqual testName expected result

testGetOutputWidthLayers :: String
testGetOutputWidthLayers =
	let
		testName = "getOutputWidth - layers"

		inputWidth = 4
		layers =
			[ LinearLayer Nothing Nothing (replicate 8 (replicate 4 0.5)) (replicate 8 0.2)
			, NonLinearLayer Nothing Nothing relu
			, LinearLayer Nothing Nothing (replicate 5 (replicate 8 0.5)) (replicate 5 0.2)
			, NonLinearLayer Nothing Nothing relu
			]
		result = getOutputWidth inputWidth layers

		expected = 5
	in checkEqual testName expected result

testActivateLayersEmpty :: String
testActivateLayersEmpty =
	let
		testName = "activateLayers - empty"

		networkInputs = [[0.8, 0.5, -0.2]]
		weights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		biases = [-0.1, -0.7]
		layer = LinearLayer Nothing Nothing weights biases
		layers = []
		result = activateLayers networkInputs layers layer

		expectedActivations = Just [[0.73, -0.41]]
		expected =
			[ LinearLayer expectedActivations (Just networkInputs) weights biases
			]
	in checkLayersApproxEqual testName expected result

testActivateLayersFull :: String
testActivateLayersFull =
	let
		testName = "activateLayers - full"

		previousWeights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		previousBiases = [-0.1, -0.7]
		previousLayer = LinearLayer (Just [[0.73, -0.41]]) (Just [[0.8, 0.5, -0.2]]) previousWeights previousBiases
		layers = [previousLayer]
		layer = NonLinearLayer Nothing Nothing relu
		result = activateLayers [] layers layer

		expectedActivations = Just [[0.73, 0]]
		expected =
			[ previousLayer
			, NonLinearLayer expectedActivations (activations previousLayer) relu
			]
	in checkLayersApproxEqual testName expected result

testForwardPropagate :: String
testForwardPropagate =
	let
		testName = "forwardPropagate"

		costFunction = meanSquaredError
		alpha = 0.1
		weights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		biases = [-0.1, -0.7]
		layers =
			[ LinearLayer Nothing Nothing weights biases
			, NonLinearLayer Nothing Nothing relu
			]
		network = Network costFunction alpha layers
		inputs = [[0.8, 0.5, -0.2]]
		result = forwardPropagate network inputs

		expectedLayers =
			[ LinearLayer (Just [[0.73, -0.41]]) (Just inputs) weights biases
			, NonLinearLayer (Just [[0.73, 0]]) (Just [[0.73, -0.41]]) relu
			]
		expected = Network costFunction alpha expectedLayers
	in checkNetworkApproxEqual testName expected result

testCalculateNetworkError :: String
testCalculateNetworkError =
	let
		testName = "calculateNetworkError"

		network = Network meanSquaredError 0 []
		actualExpectedPairs = [([0.73, 0], [0, 1])]
		result = calculateNetworkError network actualExpectedPairs

		expected = 0.76645
	in checkApproxEqual testName expected result

testBackPropagate :: String
testBackPropagate =
	let
		testName = "backPropagate"

		costFunction = meanSquaredError
		alpha = 0.1
		linearActivations = Just [[0.73, -0.41]]
		linearInputs = Just [[0.8, 0.5, -0.2]]
		weights = [[0.9, -0.1, -0.8], [0.2, 0.5, 0.6]]
		biases = [-0.1, -0.7]
		nonLinearActivations = Just [[0.73, 0]]
		layers =
			[ LinearLayer linearActivations linearInputs weights biases
			, NonLinearLayer nonLinearActivations linearActivations relu
			]
		network = Network costFunction alpha layers
		actualExpectedPairs = [([0.73, 0], [0, 1])]
		result = backPropagate network actualExpectedPairs

		expectedWeights = [[0.7832, -0.173, -0.7708], [0.2, 0.5, 0.6]]
		expectedBiases = [-0.246, -0.7]
		expectedLayers =
			[ LinearLayer Nothing Nothing expectedWeights expectedBiases
			, NonLinearLayer Nothing Nothing relu
			]
		expected = Network costFunction alpha expectedLayers
	in checkNetworkApproxEqual testName expected result

testModuleNetwork =
	[ testGetOutputWidthEmpty
	, testGetOutputWidthLayers
	, testActivateLayersEmpty
	, testActivateLayersFull
	, testForwardPropagate
	, testCalculateNetworkError
	, testBackPropagate
	]

main = do
	mapM putStrLn testModuleNetwork
