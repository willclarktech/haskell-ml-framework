module Math where

type Vector = [Float]
type Matrix = [[Float]]

type NonLinearFunction = Float -> Float
type CostFunction = ([Float], [Float]) -> Float

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

mean :: [Float] -> Float
mean ns = sum ns / fromIntegral (length ns)

squaredError :: (Float, Float) -> Float
squaredError (expected, actual) = (** 2) $ actual - expected

meanSquaredError :: CostFunction
meanSquaredError (expected, actual) = mean $ map squaredError $ zipWith (\a b -> (a, b)) expected actual
