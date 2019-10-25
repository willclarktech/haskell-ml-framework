type Activation = Float
type Weight = Float
type Bias = Float

forwardPropagateInput :: Activation -> Activation
forwardPropagateInput input =
	let
		weight = 2.5 :: Weight
		bias = (- 4.0) :: Bias
		linearLayer = input * weight + bias
	in
		linearLayer
