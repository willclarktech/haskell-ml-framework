module Math where

import Data.List

type Vector = [Float]
type Matrix = [[Float]]

deepMap :: (a -> b) -> [[a]] -> [[b]]
deepMap = map . map

groupByCondition :: (a -> Bool) -> [a] -> ([a], [a])
groupByCondition condition =
	let fn candidate (yes, no) = if condition candidate
		then (candidate:yes, no)
		else (yes, candidate:no)
	in foldr fn ([], [])

weightedSum :: [Float] -> [Float] -> Float
weightedSum input = sum . (zipWith (*) input)

vectorMatrixMultiplication :: Vector -> Matrix -> Vector
vectorMatrixMultiplication = map . weightedSum

matrixMultiplication :: Matrix -> Matrix -> Matrix
matrixMultiplication matrix1 matrix2 = map (\row -> vectorMatrixMultiplication row matrix2) matrix1

mean :: [Float] -> Float
mean ns = sum ns / fromIntegral (length ns)

costFunctionShowPrefix :: String
costFunctionShowPrefix = "CostFunction: <"

costFunctionShowSuffix :: Char
costFunctionShowSuffix = '>'

data CostFunction = CostFunction
	{ costFunctionName :: String
	, costFunctionCalculate :: ([Float], [Float]) -> Float
	, costFunctionDerivative :: ([Float], [Float]) -> [Float]
	}

instance Show CostFunction where
	show (CostFunction name _ _) = costFunctionShowPrefix ++ name ++ [costFunctionShowSuffix]

instance Read CostFunction where
	readsPrec p (' ':str) = readsPrec p str
	readsPrec _ str = case stripPrefix costFunctionShowPrefix str of
		Just body ->
			let (name, rest) = span (/= costFunctionShowSuffix) body
			in [(resolveCostFunction name, tail rest)]
		Nothing -> error "String does not represent a CostFunction"

instance Eq CostFunction where
	(CostFunction name1 _ _) == (CostFunction name2 _ _) = name1 == name2

squaredError :: Float -> Float -> Float
squaredError actual = (** 2) . (actual -)

squaredErrorDerivative :: Float -> Float -> Float
squaredErrorDerivative actual = (2 *) . (actual -)

meanSquaredError :: CostFunction
meanSquaredError =
	let
		calculate = mean . (uncurry (zipWith squaredError))
		derivative = uncurry (zipWith squaredErrorDerivative)
	in CostFunction "MSE" calculate derivative

costFunctions :: [CostFunction]
costFunctions = [meanSquaredError]

resolveCostFunction :: String -> CostFunction
resolveCostFunction requestedName =
	let result = find ((requestedName ==) . costFunctionName) costFunctions
	in case result of
		Just costFunction -> costFunction
		Nothing -> error "Cost function not found"

nonLinearFunctionShowPrefix :: String
nonLinearFunctionShowPrefix = "NonLinearFunction: <"

nonLinearFunctionShowSuffix :: Char
nonLinearFunctionShowSuffix = '>'

data NonLinearFunction = NonLinearFunction
	{ nonLinearName :: String
	, nonLinearCalculate :: Float -> Float
	, nonLinearDerivative :: Float -> Float
	}

instance Show NonLinearFunction where
	show (NonLinearFunction name _ _) = nonLinearFunctionShowPrefix ++ name ++ [nonLinearFunctionShowSuffix]

instance Read NonLinearFunction where
	readsPrec p (' ':str) = readsPrec p str
	readsPrec _ str = case stripPrefix nonLinearFunctionShowPrefix str of
		Just body ->
			let (name, rest) = span (/= nonLinearFunctionShowSuffix) body
			in [(resolveNonLinearFunction name, tail rest)]
		Nothing -> error "String does not represent a NonLinearFunction"

instance Eq NonLinearFunction where
	(NonLinearFunction name1 _ _) == (NonLinearFunction name2 _ _) = name1 == name2

relu :: NonLinearFunction
relu =
	let
		calculate = max 0
		derivative = fromIntegral . fromEnum . (> 0)
	in NonLinearFunction "relu" calculate derivative

sigmoid :: NonLinearFunction
sigmoid =
	let
		calculate = (1 /) . (1 +) . exp . (0 -)
		derivative n = n * (1 - n)
	in NonLinearFunction "sigmoid" calculate derivative

nonLinearFunctions :: [NonLinearFunction]
nonLinearFunctions = [relu, sigmoid]

resolveNonLinearFunction :: String -> NonLinearFunction
resolveNonLinearFunction requestedName =
	let result = find ((requestedName ==) . nonLinearName) nonLinearFunctions
	in case result of
		Just nonLinearFunction -> nonLinearFunction
		Nothing -> error "Non-linear function not found"
