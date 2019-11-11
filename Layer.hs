module Layer where

import Data.List
import System.Random

import Math

type Width = Int

type Activation = Float
type LayerInput = [Activation]
type LayerOutput = [Activation]

type Weight = Float
type WeightRow = [Weight]
type WeightMatrix = [WeightRow]

type Bias = Float
type BiasRow = [Bias]

type Error = Float
type LayerError = [Error]

type Update = Float
type RowUpdate = [Update]
type MatrixUpdate = [RowUpdate]

type Alpha = Float

data Layer =
	LinearLayer
		{ activations :: Maybe [LayerOutput]
		, inputs :: Maybe [LayerInput]
		, weights :: WeightMatrix
		, biases :: BiasRow
		}
	| NonLinearLayer
		{ activations :: Maybe [LayerOutput]
		, inputs :: Maybe [LayerInput]
		, nonLinearFunction :: NonLinearFunction
		}
	| NormalizationLayer
		{ activations :: Maybe [LayerOutput]
		, inputs :: Maybe [LayerInput]
		, normalizationFunction :: NormalizationFunction
		}
	deriving (Eq, Read, Show)

data LayerSpecification =
	LinearLayerSpecification Width
	| NonLinearLayerSpecification String
	| NormalizationLayerSpecification String

getRandomValues :: StdGen -> [Float]
getRandomValues = randomRs (-1.0, 1.0)

initializeBiases :: StdGen -> Width -> BiasRow
initializeBiases g width = take width $ getRandomValues g

initializeWeights :: StdGen -> Width -> Width -> WeightMatrix
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

createNormalizationLayer :: String -> Layer
createNormalizationLayer = NormalizationLayer Nothing Nothing . resolveNormalizationFunction

createLayer :: StdGen -> Width -> LayerSpecification -> Layer
createLayer g previousWidth (LinearLayerSpecification width) = createLinearLayer g previousWidth width
createLayer _ _ (NonLinearLayerSpecification name) = createNonLinearLayer name
createLayer _ _ (NormalizationLayerSpecification name) = createNormalizationLayer name

activateLayer :: [LayerInput] -> Layer -> Layer
activateLayer inputs (LinearLayer _ _ weights biases) =
	let
		weightedSums = matrixMultiplication inputs weights
		activations = map (zipWith (+) biases) weightedSums
	in LinearLayer (Just activations) (Just inputs) weights biases
activateLayer inputs (NonLinearLayer _ _ function) =
	let activations = (deepMap $ nonLinearCalculate function) inputs
	in NonLinearLayer (Just activations) (Just inputs) function
activateLayer inputs (NormalizationLayer _ _ function) =
	let activations = (map $ normalizationCalculate function) inputs
	in NormalizationLayer (Just activations) (Just inputs) function

calculateWeightUpdateOneToOne :: Error -> Activation -> Update
calculateWeightUpdateOneToOne = (*)

calculateWeightUpdateOneToN :: Error -> LayerInput -> RowUpdate
calculateWeightUpdateOneToN err = map (calculateWeightUpdateOneToOne err)

calculateWeightUpdateNToN :: LayerError -> LayerInput -> MatrixUpdate
calculateWeightUpdateNToN errors input =
	let calculatePartialWeightUpdates err = calculateWeightUpdateOneToN err input
	in map calculatePartialWeightUpdates errors

combineWeightUpdates :: MatrixUpdate -> MatrixUpdate -> MatrixUpdate
combineWeightUpdates newUpdates [] = newUpdates
combineWeightUpdates newUpdates previousUpdates = zipWith (zipWith (+)) newUpdates previousUpdates

calculateWeightUpdates :: [LayerError] -> [LayerInput] -> MatrixUpdate
calculateWeightUpdates errors inputs =
	let isolatedWeightUpdates = zipWith calculateWeightUpdateNToN errors inputs
	in foldr combineWeightUpdates [] isolatedWeightUpdates

updateWeight :: Alpha -> Weight -> Update -> Weight
updateWeight alpha weight update = weight - (update * alpha)

updateWeights :: Alpha -> WeightMatrix -> MatrixUpdate -> WeightMatrix
updateWeights alpha = zipWith (zipWith (updateWeight alpha))

updateBias :: Alpha -> Float -> Bias -> Bias
updateBias alpha bias update = bias - (update * alpha)

updateBiases :: Alpha -> BiasRow -> RowUpdate -> BiasRow
updateBiases alpha = zipWith (updateBias alpha)

calculateBiasUpdates :: [LayerError] -> RowUpdate
calculateBiasUpdates = (map mean) . transpose

calculateNextLinearLayerErrorOneToOne :: Error -> Weight -> Error
calculateNextLinearLayerErrorOneToOne = (*)

calculateNextLinearLayerErrorOneToN :: Error -> WeightRow -> LayerError
calculateNextLinearLayerErrorOneToN error = map (calculateNextLinearLayerErrorOneToOne error)

calculateNextLinearLayerErrorNToN :: WeightMatrix -> LayerError -> LayerError
calculateNextLinearLayerErrorNToN weightsList errors =
	let perNodeErrors = zipWith calculateNextLinearLayerErrorOneToN errors weightsList
	in map sum $ transpose perNodeErrors

calculateNextLinearLayerErrors :: WeightMatrix -> [LayerError] -> [LayerError]
calculateNextLinearLayerErrors weights = map (calculateNextLinearLayerErrorNToN weights)

calculateNextNonLinearLayerErrors :: [LayerError] -> [[Float]] -> [LayerError]
calculateNextNonLinearLayerErrors = zipWith (zipWith (*))

calculateNextNormalizationLayerErrors :: [LayerError] -> [[Float]] -> [LayerError]
calculateNextNormalizationLayerErrors = zipWith (zipWith (*))

updateLayer :: Alpha -> Layer -> [LayerError] -> (Layer, [LayerError])
updateLayer alpha (LinearLayer activations (Just inputs) weights biases) errors =
	let
		weightUpdates = calculateWeightUpdates errors inputs
		newWeights = updateWeights alpha weights weightUpdates
		biasUpdates = calculateBiasUpdates errors
		newBiases = updateBiases alpha biases biasUpdates
		newErrors = calculateNextLinearLayerErrors weights errors
	in (LinearLayer Nothing Nothing newWeights newBiases, newErrors)
updateLayer _ (NonLinearLayer (Just activations) (Just inputs) function) errors =
	let
		derivatives = deepMap (nonLinearDerivative function) activations
		newErrors = calculateNextNonLinearLayerErrors errors derivatives
	in (NonLinearLayer Nothing Nothing function, newErrors)
updateLayer _ (NormalizationLayer (Just activations) (Just inputs) function) errors =
	let
		derivatives = map (normalizationDerivative function) activations
		weightedDerivatives = zipWith (zipWith (\e -> map (* e))) errors derivatives
		newErrors = deepMap sum $ map transpose weightedDerivatives
	in (NormalizationLayer Nothing Nothing function, newErrors)
updateLayer _ _ _ = error "Cannot update non-activated layer"

updateNextLayer :: Alpha -> Layer -> ([LayerError], [Layer]) -> ([LayerError], [Layer])
updateNextLayer alpha layer (errors, previousLayers) =
	let (updatedLayer, newErrors) = updateLayer alpha layer errors
	in (newErrors, updatedLayer : previousLayers)
