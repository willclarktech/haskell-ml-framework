module Lib where

type Vector = [Float]
type Matrix = [[Float]]

type Activation = Float
type Weight = Float
type Bias = Float
type Width = Int

type NonLinearFunction = Float -> Float

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

sigmoid :: NonLinearFunction
sigmoid = (1 /) . (1 +) . exp . (0 -)

relu :: NonLinearFunction
relu n = if n > 0 then n else 0

weightedSum :: [Activation] -> [Weight] -> Activation
weightedSum input = sum . (zipWith (*) input)

vectorMatrixMultiplication :: Vector -> Matrix -> Vector
vectorMatrixMultiplication = map . weightedSum

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
