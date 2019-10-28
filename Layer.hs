module Layer where

import System.Random
import Math

type Weight = Float
type Bias = Float
type Width = Int
type Activation = Float
type Input = [Activation]
type Output = [Activation]

data Layer =
	LinearLayer
		{ activations :: Maybe [Output]
		, weights :: [[Weight]]
		, biases :: [Bias]
		}
	| NonLinearLayer
		{ activations :: Maybe [Output]
		, function :: NonLinearFunction
		}
	deriving (Show)

data LayerSpecification =
	LinearLayerSpecification Width
	| NonLinearLayerSpecification String

activateLinearLayer :: Layer -> [Input] -> Layer
activateLinearLayer (LinearLayer _ weights biases) inputs =
	let
		weightedSums = matrixMultiplication inputs weights
		activations = map (zipWith (+) biases) weightedSums
	in LinearLayer (Just activations) weights biases
activateLinearLayer _ _ = error "Cannot activate non-linear layer"

activateNonLinearLayer :: Layer -> [Input] -> Layer
activateNonLinearLayer (NonLinearLayer _ function) inputs =
	let activations = (map $ map $ nonLinearCalculate function) inputs
	in NonLinearLayer (Just activations) function
activateNonLinearLayer _ _ = error "Cannot activate linear layer"

activateLayer :: [Input] -> Layer -> Layer
activateLayer inputs layer =
	case layer of
		LinearLayer _ _ _ -> activateLinearLayer layer inputs
		NonLinearLayer _ _ -> activateNonLinearLayer layer inputs

getRandomValues :: StdGen -> [Float]
getRandomValues = randomRs (-1.0, 1.0)

initializeBiases :: StdGen -> Width -> [Bias]
initializeBiases g width = take width $ getRandomValues g

initializeWeights :: StdGen -> Width -> Width -> [[Weight]]
initializeWeights g previousWidth width =
	let
		randomValues = take (previousWidth * width) $ getRandomValues g
		foldFn (ws, rs) _ =
			let (rsNow, rsNext) = splitAt previousWidth rs
			in (rsNow:ws, rsNext)
		(weights, _) = foldl foldFn ([], randomValues) (replicate width 0)
	in weights

createLinearLayer :: StdGen -> Int -> Int -> Layer
createLinearLayer g previousWidth width =
	let (g1, g2) = split g
	in LinearLayer Nothing (initializeWeights g1 previousWidth width) (initializeBiases g2 width)

createNonLinearLayer :: String -> Layer
createNonLinearLayer = NonLinearLayer Nothing . resolveNonLinearFunction

createLayer :: StdGen -> Width -> LayerSpecification -> Layer
createLayer g previousWidth (LinearLayerSpecification width) = createLinearLayer g previousWidth width
createLayer _ _ (NonLinearLayerSpecification name) = createNonLinearLayer name

updateNextLayer :: Layer -> ([Float], [Layer]) -> ([Float], [Layer])
updateNextLayer (LinearLayer (Just activations) weights biases) (errors, previousLayers) =
	let
		meanActivations = map mean $ transpose activations
		newWeights = zipWith (\e ws -> zipWith (\w a -> w - (e * a)) ws meanActivations) errors weights
		newBiases = zipWith (-) biases errors
		newErrors = errors
		updatedLayer = LinearLayer Nothing newWeights newBiases
	in (newErrors, updatedLayer : previousLayers)
-- updateNextLayer layer (errors, previousLayers) = (errors, layer : previousLayers)
updateNextLayer _ _ = error "Layer type not supported"
