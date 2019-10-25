module Lib where

type Activation = Float
type Weight = Float
type Bias = Float

type NonLinearFunction = Float -> Float

data Layer =
	LinearLayer
		{ weight :: Weight
		, bias :: Bias
		}
	| NonLinearLayer
		{ function :: NonLinearFunction
		}

type Network = [Layer]

sigmoid :: NonLinearFunction
sigmoid = (1 /) . (1 +) . exp . (0 -)

applyLinearLayer :: Layer -> Activation -> Activation
applyLinearLayer (LinearLayer weight bias) = (+ bias) . (* weight)
applyLinearLayer _ = error "Cannot apply non-linear layer"

applyNonLinearLayer :: Layer -> Activation -> Activation
applyNonLinearLayer (NonLinearLayer function) = function
applyNonLinearLayer _ = error "Cannot apply linear layer"

applyLayer :: Activation -> Layer -> Activation
applyLayer input layer =
	case layer of
		LinearLayer _ _ -> applyLinearLayer layer input
		NonLinearLayer _ -> applyNonLinearLayer layer input

forwardPropagateInput :: Network -> Activation -> Activation
forwardPropagateInput network input = foldl applyLayer input network
