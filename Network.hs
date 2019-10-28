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
		LinearLayer weights _ _ -> length weights
		NonLinearLayer _ _ -> getOutputWidth inputWidth $ init layers

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

getFinalActivations :: [Layer] -> [Output]
getFinalActivations layers = case activations $ last layers of
	Nothing -> error "Final layer not activated"
	Just a -> a

getOutputs :: Network -> [Output]
getOutputs = getFinalActivations . layers

activateLayers :: [Input] -> [Layer] -> Layer -> [Layer]
activateLayers networkInputs ls layer =
	let layerInputs = if length ls == 0 then networkInputs else getFinalActivations ls
	in activateLayer layerInputs layer : ls

forwardPropagate :: Network -> [Input] -> Network
forwardPropagate (Network layers costFunction) inputs =
	let activatedLayers = foldl (activateLayers inputs) [] $ layers
	in Network activatedLayers costFunction

backPropagate :: Network -> [(Output, Output)] -> Network
backPropagate (Network layers costFunction) actualExpectedPairs =
	let
		errs = map (uncurry (zipWith (\a b -> 2 * (a - b)))) actualExpectedPairs
		averagedErrs = map mean $ transpose errs
		newLayers = snd $ foldr updateLayer (averagedErrs, []) layers
	in Network newLayers costFunction

runIteration :: Network -> [Input] -> [Output] -> (Network, Float)
runIteration network inputs expectedOutputs =
	let
		activatedNetwork = forwardPropagate network inputs
		outputs = getOutputs activatedNetwork
		actualExpectedPairs = zip outputs expectedOutputs
		err = mean $ map (costFunctionCalculate (costFunction network)) $ actualExpectedPairs
		trained = backPropagate network actualExpectedPairs
	in (trained, err)
