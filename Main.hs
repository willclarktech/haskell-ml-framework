type Activation = Float
type Weight = Float
type Bias = Float

applyLinearLayer :: Weight -> Bias -> Activation -> Activation
applyLinearLayer weight bias = (+ bias) . (* weight)

applyNonLinearLayer :: Activation -> Activation
applyNonLinearLayer = (1 /) . (1 +) . exp . (0 -)

forwardPropagateInput :: Activation -> Activation
forwardPropagateInput input =
	let
		weight = 2.5 :: Weight
		bias = (- 4.0) :: Bias
		linearLayer = applyLinearLayer weight bias input
		nonLinearLayer = applyNonLinearLayer linearLayer
	in
		nonLinearLayer
