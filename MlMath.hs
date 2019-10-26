module MlMath where

type Vector = [Float]
type Matrix = [[Float]]

type NonLinearFunction = Float -> Float

sigmoid :: NonLinearFunction
sigmoid = (1 /) . (1 +) . exp . (0 -)

relu :: NonLinearFunction
relu n = if n > 0 then n else 0

resolveNonLinearFunction :: String -> NonLinearFunction
resolveNonLinearFunction name =
	case name of
		"sigmoid" -> sigmoid
		"relu" -> relu
		_ -> error "Non-linear function not supported"

weightedSum :: [Float] -> [Float] -> Float
weightedSum input = sum . (zipWith (*) input)

vectorMatrixMultiplication :: Vector -> Matrix -> Vector
vectorMatrixMultiplication = map . weightedSum
