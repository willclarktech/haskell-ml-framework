type Activation = Float
type Weight = Float
type Bias = Float

applyLinearLayer :: Weight -> Bias -> Activation -> Activation
applyLinearLayer weight bias = (+ bias) . (* weight)

forwardPropagateInput :: Activation -> Activation
forwardPropagateInput input =
	let
		weight = 2.5 :: Weight
		bias = (- 4.0) :: Bias
		linearLayer = applyLinearLayer weight bias input
		nonLinearLayer = 1 / (1 + exp (-linearLayer))
	in
		nonLinearLayer
