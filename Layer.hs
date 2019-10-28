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
		{ weights :: [[Weight]]
		, biases :: [Bias]
		}
	| NonLinearLayer
		{ function :: NonLinearFunction
		}
	deriving (Show)

data LayerSpecification =
	LinearLayerSpecification Width
	| NonLinearLayerSpecification String

applyLinearLayer :: Layer -> [Input] -> [Output]
applyLinearLayer (LinearLayer weights biases) inputs =
	let weightedSums = matrixMultiplication inputs weights
	in map (zipWith (+) biases) weightedSums
applyLinearLayer _ _ = error "Cannot apply non-linear layer"

applyNonLinearLayer :: Layer -> [Input] -> [Output]
applyNonLinearLayer (NonLinearLayer function) = map $ map $ nonLinearCalculate function
applyNonLinearLayer _ = error "Cannot apply linear layer"

applyLayer :: [Input] -> Layer -> [Output]
applyLayer inputs layer =
	case layer of
		LinearLayer _ _ -> applyLinearLayer layer inputs
		NonLinearLayer _ -> applyNonLinearLayer layer inputs

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
	in LinearLayer (initializeWeights g1 previousWidth width) (initializeBiases g2 width)

createNonLinearLayer :: String -> Layer
createNonLinearLayer = NonLinearLayer . resolveNonLinearFunction

createLayer :: StdGen -> Width -> LayerSpecification -> Layer
createLayer g previousWidth (LinearLayerSpecification width) = createLinearLayer g previousWidth width
createLayer _ _ (NonLinearLayerSpecification name) = createNonLinearLayer name

updateLayer :: Layer -> ([Float], [Layer]) -> ([Float], [Layer])
updateLayer (LinearLayer weights biases) (errors, previousLayers) =
	let
		newWeights = zipWith (\error -> map (\w -> w - error)) errors weights
		newBiases = zipWith (-) biases errors
		newErrors = errors
		updatedLayer = LinearLayer newWeights newBiases
	in (newErrors, updatedLayer : previousLayers)
updateLayer layer (errors, previousLayers) = (errors, layer : previousLayers)
