type Activation = Float
type Weight = Float
type Bias = Float

sigmoid :: Float -> Float
sigmoid = (1 /) . (1 +) . exp . (0 -)

applyLinearLayer :: Weight -> Bias -> Activation -> Activation
applyLinearLayer weight bias = (+ bias) . (* weight)

applyNonLinearLayer :: Activation -> Activation
applyNonLinearLayer = sigmoid

forwardPropagateInput :: Activation -> Activation
forwardPropagateInput input =
	let
		weight = 2.5 :: Weight
		bias = (- 4.0) :: Bias
		linearLayer = applyLinearLayer weight bias input
		nonLinearLayer = applyNonLinearLayer linearLayer
	in
		nonLinearLayer
