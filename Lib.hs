module Lib where

type Activation = Float
type Weight = Float
type Bias = Float

type NonLinearFunction = Float -> Float

data Layer = Layer
	{ weight :: Weight
	, bias :: Bias
	}

sigmoid :: NonLinearFunction
sigmoid = (1 /) . (1 +) . exp . (0 -)

applyLinearLayer :: Layer -> Activation -> Activation
applyLinearLayer (Layer bias weight) = (+ bias) . (* weight)

applyNonLinearLayer :: Activation -> Activation
applyNonLinearLayer = sigmoid

forwardPropagateInput :: Activation -> Activation
forwardPropagateInput input =
	let
		weight = 2.5 :: Weight
		bias = (- 4.0) :: Bias
		linearLayer = Layer weight bias
		linearLayerActivation = applyLinearLayer linearLayer input
		nonLinearLayerActivation = applyNonLinearLayer linearLayerActivation
	in
		nonLinearLayerActivation
