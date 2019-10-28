module Math where

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
	, nonLinearDerivative :: Float -> Float
	}
instance Show NonLinearFunction where
	show (NonLinearFunction name _ _) = "NonLinearFunction: " ++ name

find :: Foldable t => (a -> Bool) -> t a -> Maybe a
find condition =
	let fn candidate result
		| condition candidate = Just candidate
		| otherwise = result
	in foldr fn Nothing

sigmoid :: NonLinearFunction
sigmoid =
	let
		calculate = (1 /) . (1 +) . exp . (0 -)
		derivative n = n * (1 - n)
	in NonLinearFunction "sigmoid" calculate derivative

relu :: NonLinearFunction
relu =
	let
		calculate n = if n > 0 then n else 0
		derivative n = if n > 0 then 1 else 0
	in NonLinearFunction "relu" calculate derivative

nonLinearFunctions :: [NonLinearFunction]
nonLinearFunctions = [sigmoid, relu]

resolveNonLinearFunction :: String -> NonLinearFunction
resolveNonLinearFunction requestedName =
	let result = find ((requestedName ==) . nonLinearName) nonLinearFunctions
	in case result of
		Just nonLinearFunction -> nonLinearFunction
		Nothing -> error "Non-linear function not found"

transpose :: Matrix -> Matrix
transpose [] = []
transpose ([]:_) = []
transpose matrix = (map head matrix) : (transpose $ map tail matrix)

weightedSum :: [Float] -> [Float] -> Float
weightedSum input = sum . (zipWith (*) input)

vectorMatrixMultiplication :: Vector -> Matrix -> Vector
vectorMatrixMultiplication = map . weightedSum

matrixMultiplication :: Matrix -> Matrix -> Matrix
matrixMultiplication matrix1 matrix2 =
    let transposedMatrix = transpose matrix2
    in map (\row -> vectorMatrixMultiplication row transposedMatrix) matrix1

mean :: [Float] -> Float
mean ns = sum ns / fromIntegral (length ns)

squaredError :: (Float, Float) -> Float
squaredError (actual, expected) = (** 2) $ actual - expected

meanSquaredError :: CostFunction
meanSquaredError =
	let fn = mean . (map squaredError) . (uncurry zip)
	in CostFunction "MSE" fn
