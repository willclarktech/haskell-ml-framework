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
		, inputs :: Maybe [Input]
		, function :: NonLinearFunction
		}
	deriving (Show, Eq)

data LayerSpecification =
	LinearLayerSpecification Width
	| NonLinearLayerSpecification String

activateLayer :: [Input] -> Layer -> Layer
activateLayer inputs (NonLinearLayer _ _ function) =
	let activations = (map $ map $ nonLinearCalculate function) inputs
	in NonLinearLayer (Just activations) (Just inputs) function
activateLayer inputs (LinearLayer _ _ weights biases) =
	let
		weightedSums = matrixMultiplication inputs weights
		activations = map (zipWith (+) biases) weightedSums
	in LinearLayer (Just activations) (Just inputs) weights biases

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
createNonLinearLayer = NonLinearLayer Nothing Nothing . resolveNonLinearFunction

createLayer :: StdGen -> Width -> LayerSpecification -> Layer
createLayer g previousWidth (LinearLayerSpecification width) = createLinearLayer g previousWidth width
createLayer _ _ (NonLinearLayerSpecification name) = createNonLinearLayer name

updateWeight :: Float -> Float -> Activation -> Weight -> Weight
updateWeight alpha err activation weight = weight - (err * activation * alpha)

updatePartialWeights :: Float -> Input -> Float -> [Weight] -> [Weight]
updatePartialWeights alpha input err = zipWith (updateWeight alpha err) input

updateBias :: Float -> Float -> Bias -> Bias
updateBias alpha err bias = bias - (err * alpha)

calculateMeanInput :: [Input] -> Input
calculateMeanInput = map mean . transpose

calculateNextErrors :: [[Weight]] -> [Float] -> [Float]
calculateNextErrors weights errors = map sum $ transpose $ zipWith (\e -> map (* e) ) errors weights

updateLayer :: Float -> Layer -> [Float] -> (Layer, [Float])
updateLayer alpha (LinearLayer _ (Just inputs) weights biases) errors =
	let
		meanInput = calculateMeanInput inputs
		newWeights = zipWith (updatePartialWeights alpha meanInput) errors weights
		newBiases = zipWith (updateBias alpha) errors biases
		newErrors = calculateNextErrors weights errors
	in (LinearLayer Nothing Nothing newWeights newBiases, newErrors)
updateLayer _ (NonLinearLayer _ (Just inputs) function) errors =
	let
		meanInput = calculateMeanInput inputs
		newErrors = zipWith (\i e -> e * (nonLinearDerivative function $ i)) meanInput errors
	in (NonLinearLayer Nothing Nothing function, newErrors)
updateLayer _ _ _ = error "Cannot update non-activated layer"

updateNextLayer :: Float -> Layer -> ([Float], [Layer]) -> ([Float], [Layer])
updateNextLayer alpha layer (errors, previousLayers) =
	let (updatedLayer, newErrors) = updateLayer alpha layer errors
	in (newErrors, updatedLayer : previousLayers)
