module Network where

import Debug.Trace
import System.Random

import Layer
import Math

type NetworkError = Float

data Network = Network
	{ costFunction :: CostFunction
	, alpha :: Alpha
	, layers :: [Layer]
	}
	deriving (Read, Show)

getOutputWidth :: Width -> [Layer] -> Width
getOutputWidth inputWidth [] = inputWidth
getOutputWidth inputWidth layers =
	case last layers of
		LinearLayer _ _ weights _ -> length weights
		NonLinearLayer _ _ _ -> getOutputWidth inputWidth $ init layers

appendLayer :: Width -> (StdGen, [Layer]) -> LayerSpecification -> (StdGen, [Layer])
appendLayer inputWidth (g, layers) specification =
	let
		previousWidth = getOutputWidth inputWidth layers
		newLayer = createLayer g previousWidth specification
		newG = snd $ next g
	in (newG, layers ++ [newLayer])

createNetwork :: StdGen -> Alpha -> Width -> [LayerSpecification] -> Network
createNetwork g alpha inputWidth layerSpecifications =
	let (_, layers) = foldl (appendLayer inputWidth) (g, []) layerSpecifications
	in Network meanSquaredError alpha layers

getFinalActivations :: [Layer] -> [LayerOutput]
getFinalActivations [] = error "No layers"
getFinalActivations layers = case activations $ last layers of
	Nothing -> error "Final layer not activated"
	Just a -> a

getOutputs :: Network -> [LayerOutput]
getOutputs = getFinalActivations . layers

activateLayers :: [LayerInput] -> [Layer] -> Layer -> [Layer]
activateLayers networkInputs ls layer =
	let layerInputs = if length ls == 0 then networkInputs else getFinalActivations ls
	in ls ++ [activateLayer layerInputs layer]

forwardPropagate :: Network -> [LayerInput] -> Network
forwardPropagate (Network costFunction alpha layers) inputs =
	let activatedLayers = foldl (activateLayers inputs) [] layers
	in Network costFunction alpha activatedLayers

calculateNetworkError :: Network -> [(LayerOutput, LayerOutput)] -> NetworkError
calculateNetworkError (Network costFunction _ _) = mean . (map (costFunctionCalculate costFunction))

backPropagate :: Network -> [(LayerOutput, LayerOutput)] -> Network
backPropagate (Network costFunction alpha layers) actualExpectedPairs =
	let
		errs = map (costFunctionDerivative costFunction) actualExpectedPairs
		newLayers = snd $ foldr (updateNextLayer alpha) (errs, []) layers
	in Network costFunction alpha newLayers

-- divideInputsIntoMiniBatches :: Int -> [LayerInput] -> [LayerOutput] -> [MiniBatch]
-- divideInputsIntoMiniBatches size [] _ = []
-- divideInputsIntoMiniBatches size _ [] = []
-- divideInputsIntoMiniBatches size inputs outputs =
-- 	let
-- 		(miniBatchInputs, remainingInputs) = splitAt size inputs
-- 		(miniBatchOutputs, remainingOutputs) = splitAt size outputs
-- 	in (miniBatchInputs, miniBatchOutputs) : divideInputsIntoMiniBatches size remainingInputs remainingOutputs

runIteration :: Network -> [LayerInput] -> [LayerOutput] -> (Network, NetworkError)
runIteration network inputs expectedOutputs =
	let
		activatedNetwork = forwardPropagate network inputs
		outputs = getOutputs activatedNetwork
		actualExpectedPairs = zip outputs expectedOutputs
		err = calculateNetworkError activatedNetwork actualExpectedPairs
		trained = backPropagate activatedNetwork actualExpectedPairs
	in (trained, err)

-- runMiniBatch :: MiniBatch -> (Network, NetworkError) -> (Network, NetworkError)
-- runMiniBatch (inputs, expectedOutputs) (network, _) = runIteration network inputs expectedOutputs

-- runIterationWithMiniBatches :: Network -> [MiniBatch] -> (Network, NetworkError)
-- runIterationWithMiniBatches network = foldr runMiniBatch (network, 0.0)

run :: Int -> [LayerInput] -> [LayerOutput] -> Network -> Int -> (Network, NetworkError)
run logCycleSize _ _ _ n
	| n < 0 = error "Iterations cannot be negative"
	| logCycleSize < 0 = error "Log cycle size cannot be negative"
run logCycleSize inputs expectedOutputs network 0 =
	let
		activatedNetwork = forwardPropagate network inputs
		outputs = getOutputs activatedNetwork
		actualExpectedPairs = zip outputs expectedOutputs
		err = calculateNetworkError activatedNetwork actualExpectedPairs
	in (activatedNetwork, err)
run logCycleSize inputs expectedOutputs network n =
	let
		shouldLogError = if logCycleSize == 0 then False else 0 == mod n logCycleSize
		(trained, err) = runIteration network inputs expectedOutputs
	in if shouldLogError && trace ("Iteration: -" ++ show n ++ "; Error: " ++ show err) False then (trained, err) else case n of
		1 -> (trained, err)
		_ -> run logCycleSize inputs expectedOutputs trained (n - 1)

writeNetworkToFile :: FilePath -> Network -> IO ()
writeNetworkToFile filePath = (writeFile filePath) . show

readNetworkFromFile :: FilePath -> IO Network
readNetworkFromFile filePath = do
	contents <- readFile filePath
	let network = read contents :: Network
	return network
