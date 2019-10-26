module Math where

import Data.List

type Vector = [Float]
type Matrix = [[Float]]

data CostFunction = CostFunction
	{ costFunctionName :: String
	, costFunctionCalculate :: ([Float], [Float]) -> Float
	}
instance Show CostFunction where
	show (CostFunction name _) = "CostFunction: " ++ name
data NonLinearFunction = NonLinearFunction
	{ nonLinearName :: String
	, nonLinearCalculate :: Float -> Float
	}
instance Show NonLinearFunction where
	show (NonLinearFunction name _) = "NonLinearFunction: " ++ name

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
	let fn = mean . (map squaredError) . (uncurry zip)
	in CostFunction "MSE" fn
