module Network where

import System.Random
import Layer
import Math

data Network = Network
	{ layers :: [Layer]
	, costFunction :: CostFunction
	}
	deriving (Show)

getOutputWidth :: Width -> [Layer] -> Width
getOutputWidth inputWidth [] = inputWidth
getOutputWidth inputWidth layers =
	case last layers of
		LinearLayer weights _ -> length weights
		NonLinearLayer _ -> getOutputWidth inputWidth $ init layers

appendLayer :: Width -> (StdGen, [Layer]) -> LayerSpecification -> (StdGen, [Layer])
appendLayer inputWidth (g, layers) specification =
	let
		previousWidth = getOutputWidth inputWidth layers
		newLayer = createLayer g previousWidth specification
		newG = snd $ next g
	in (newG, layers ++ [newLayer])

createNetwork :: StdGen -> Width -> [LayerSpecification] -> Network
createNetwork g inputWidth layerSpecifications =
	let (_, layers) = foldl (appendLayer inputWidth) (g, []) layerSpecifications
	in Network layers meanSquaredError

forwardPropagateInputs :: Network -> [Input] -> [Output]
forwardPropagateInputs network inputs = foldl applyLayer inputs $ layers network

runIteration :: Network -> [Input] -> [Output] -> (Network, Float)
runIteration network inputs expectedOutputs =
	let
		outputs = forwardPropagateInputs network inputs
		err = mean $ map (costFunctionCalculate (costFunction network)) $ zipWith (\a b -> (a, b)) expectedOutputs outputs
	in (network, err)
