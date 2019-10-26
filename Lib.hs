module Lib where

import System.Random
import MlMath

type Activation = Float
type Weight = Float
type Bias = Float
type Width = Int

data Layer =
	LinearLayer
		{ weights :: [[Weight]]
		, biases :: [Bias]
		}
	| NonLinearLayer
		{ function :: NonLinearFunction
		}

data LayerSpecification =
	LinearLayerSpecification Width
	| NonLinearLayerSpecification String

type Network = [Layer]

applyLinearLayer :: Layer -> [Activation] -> [Activation]
applyLinearLayer (LinearLayer weights biases) input =
	let weightedSums = vectorMatrixMultiplication input weights
	in zipWith (+) biases weightedSums
applyLinearLayer _ _ = error "Cannot apply non-linear layer"

applyNonLinearLayer :: Layer -> [Activation] -> [Activation]
applyNonLinearLayer (NonLinearLayer function) = map function
applyNonLinearLayer _ = error "Cannot apply linear layer"

applyLayer :: [Activation] -> Layer -> [Activation]
applyLayer input layer =
	case layer of
		LinearLayer _ _ -> applyLinearLayer layer input
		NonLinearLayer _ -> applyNonLinearLayer layer input

forwardPropagateInput :: Network -> [Activation] -> [Activation]
forwardPropagateInput network input = foldl applyLayer input network

getOutputWidth :: Width -> Network -> Width
getOutputWidth inputWidth [] = inputWidth
getOutputWidth inputWidth network =
	case last network of
		LinearLayer weights _ -> length weights
		NonLinearLayer _ -> getOutputWidth inputWidth $ init network

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

appendLayer :: Width -> (StdGen, Network) -> LayerSpecification -> (StdGen, Network)
appendLayer inputWidth (g, network) specification =
	let
		previousWidth = getOutputWidth inputWidth network
		newLayer = case specification of
			LinearLayerSpecification width -> createLinearLayer g previousWidth width
			NonLinearLayerSpecification name -> createNonLinearLayer name
		newG = snd $ next g
	in (newG, network ++ [newLayer])

createNetwork :: StdGen -> Width -> [LayerSpecification] -> Network
createNetwork g inputWidth layerSpecifications =
	let (_, network) = foldl (appendLayer inputWidth) (g, []) layerSpecifications
	in network
