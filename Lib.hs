module Lib where

type Activation = Float
type Weight = Float
type Bias = Float

type NonLinearFunction = Float -> Float

data LinearLayer = LinearLayer
	{ weight :: Weight
	, bias :: Bias
	}

data NonLinearLayer = NonLinearLayer
	{ function :: NonLinearFunction
	}

sigmoid :: NonLinearFunction
sigmoid = (1 /) . (1 +) . exp . (0 -)

applyLinearLayer :: LinearLayer -> Activation -> Activation
applyLinearLayer (LinearLayer bias weight) = (+ bias) . (* weight)

applyNonLinearLayer :: NonLinearLayer -> Activation -> Activation
applyNonLinearLayer (NonLinearLayer function) = function

forwardPropagateInput :: LinearLayer -> NonLinearLayer -> Activation -> Activation
forwardPropagateInput linearLayer nonLinearLayer input =
	let
		linearLayerActivation = applyLinearLayer linearLayer input
		nonLinearLayerActivation = applyNonLinearLayer nonLinearLayer linearLayerActivation
	in
		nonLinearLayerActivation
