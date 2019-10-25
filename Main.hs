type Activation = Float
type Weight = Float
type Bias = Float

data Layer = Layer
	{ weight :: Weight
	, bias :: Bias
	}

sigmoid :: Float -> Float
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
		layer = Layer weight bias
		linearLayer = applyLinearLayer layer input
		nonLinearLayer = applyNonLinearLayer linearLayer
	in
		nonLinearLayer
