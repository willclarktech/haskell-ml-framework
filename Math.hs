module Math where

type Vector = [Float]
type Matrix = [[Float]]

data CostFunction = CostFunction
	{ costFunctionName :: String
	, costFunctionCalculate :: ([Float], [Float]) -> Float
	, costFunctionDerivative :: ([Float], [Float]) -> [Float]
	}
instance Show CostFunction where
	show (CostFunction name _ _) = "CostFunction: " ++ name
instance Eq CostFunction where
	(CostFunction name1 _ _) == (CostFunction name2 _ _) = name1 == name2
data NonLinearFunction = NonLinearFunction
	{ nonLinearName :: String
	, nonLinearCalculate :: Float -> Float
	, nonLinearDerivative :: Float -> Float
	}
instance Show NonLinearFunction where
	show (NonLinearFunction name _ _) = "NonLinearFunction: " ++ name
instance Eq NonLinearFunction where
	(NonLinearFunction name1 _ _) == (NonLinearFunction name2 _ _) = name1 == name2

deepMap :: (a -> b) -> [[a]] -> [[b]]
deepMap = map . map

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
		calculate = max 0
		derivative = fromIntegral . fromEnum . (> 0)
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
matrixMultiplication matrix1 matrix2 = map (\row -> vectorMatrixMultiplication row matrix2) matrix1

mean :: [Float] -> Float
mean ns = sum ns / fromIntegral (length ns)

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
