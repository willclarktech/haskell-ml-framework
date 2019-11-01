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
		, inputs :: Maybe [Input]
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
activateLinearLayer (LinearLayer _ _ weights biases) inputs =
	let
		weightedSums = matrixMultiplication inputs weights
		activations = map (zipWith (+) biases) weightedSums
	in LinearLayer (Just activations) (Just inputs) weights biases
activateLinearLayer _ _ = error "Cannot activate non-linear layer"

activateNonLinearLayer :: Layer -> [Input] -> Layer
activateNonLinearLayer (NonLinearLayer _ function) inputs =
	let activations = (map $ map $ nonLinearCalculate function) inputs
	in NonLinearLayer (Just activations) function
activateNonLinearLayer _ _ = error "Cannot activate linear layer"

activateLayer :: [Input] -> Layer -> Layer
activateLayer inputs layer =
	case layer of
		LinearLayer _ _ _ _ -> activateLinearLayer layer inputs
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
	in LinearLayer Nothing Nothing (initializeWeights g1 previousWidth width) (initializeBiases g2 width)

createNonLinearLayer :: String -> Layer
createNonLinearLayer = NonLinearLayer Nothing . resolveNonLinearFunction

createLayer :: StdGen -> Width -> LayerSpecification -> Layer
createLayer g previousWidth (LinearLayerSpecification width) = createLinearLayer g previousWidth width
createLayer _ _ (NonLinearLayerSpecification name) = createNonLinearLayer name

updateLayer :: Layer -> [Float] -> (Layer, [Float])
updateLayer (LinearLayer _ (Just inputs) weights biases) errors =
	let
		meanInputs = map mean $ transpose inputs
		newWeights = zipWith (\e ws -> zipWith (\w i -> w - (e * i)) ws meanInputs) errors weights
		newBiases = zipWith (-) biases errors
		newErrors = map sum $ transpose $ zipWith (\e -> map (* e) ) errors weights
	in (LinearLayer Nothing Nothing newWeights newBiases, newErrors)
updateLayer (NonLinearLayer _ function) errors =
	let newErrors = map (nonLinearDerivative function) errors
	in (NonLinearLayer Nothing function, newErrors)
updateLayer _ _ = error "Cannot update non-activated layer"

updateNextLayer :: Layer -> ([Float], [Layer]) -> ([Float], [Layer])
updateNextLayer layer (errors, previousLayers) =
	let (updatedLayer, newErrors) = updateLayer layer errors
	in (newErrors, updatedLayer : previousLayers)
