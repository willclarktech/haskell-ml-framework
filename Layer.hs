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

applyLinearLayer :: Layer -> [Activation] -> [Activation]
applyLinearLayer (LinearLayer weights biases) input =
	let weightedSums = vectorMatrixMultiplication input weights
	in zipWith (+) biases weightedSums
applyLinearLayer _ _ = error "Cannot apply non-linear layer"

applyNonLinearLayer :: Layer -> [Activation] -> [Activation]
applyNonLinearLayer (NonLinearLayer function) = map $ nonLinearCalculate function
applyNonLinearLayer _ = error "Cannot apply linear layer"

applyLayer :: [Activation] -> Layer -> [Activation]
applyLayer input layer =
	case layer of
		LinearLayer _ _ -> applyLinearLayer layer input
		NonLinearLayer _ -> applyNonLinearLayer layer input

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
		(weights, _) = foldl foldFn ([], randomValues) (replicate previousWidth 0)
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
