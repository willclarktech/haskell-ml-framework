module Math where

import Data.List

type Vector = [Float]
type Matrix = [[Float]]

data CostFunction = CostFunction
	{ costFunctionName :: String
	, costFunctionCalculate :: ([Float], [Float]) -> Float
	}
data NonLinearFunction = NonLinearFunction
	{ nonLinearName :: String
	, nonLinearCalculate :: Float -> Float
	}

sigmoid :: NonLinearFunction
sigmoid =
	let fn = (1 /) . (1 +) . exp . (0 -)
	in NonLinearFunction "sigmoid" fn

relu :: NonLinearFunction
relu =
	let fn n = if n > 0 then n else 0
	in NonLinearFunction "relu" fn

nonLinearFunctions :: [NonLinearFunction]
nonLinearFunctions = [sigmoid, relu]

resolveNonLinearFunction :: String -> NonLinearFunction
resolveNonLinearFunction requestedName =
	let result = find ((requestedName ==) . nonLinearName) nonLinearFunctions
	in case result of
		Just nonLinearFunction -> nonLinearFunction
		Nothing -> error "Non-linear function not found"

weightedSum :: [Float] -> [Float] -> Float
weightedSum input = sum . (zipWith (*) input)

vectorMatrixMultiplication :: Vector -> Matrix -> Vector
vectorMatrixMultiplication = map . weightedSum

mean :: [Float] -> Float
mean ns = sum ns / fromIntegral (length ns)

squaredError :: (Float, Float) -> Float
squaredError (expected, actual) = (** 2) $ actual - expected

meanSquaredError :: CostFunction
meanSquaredError =
	let fn (expected, actual) = mean $ map squaredError $ zipWith (\a b -> (a, b)) expected actual
	in CostFunction "MSE" fn
