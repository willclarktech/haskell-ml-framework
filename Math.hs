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
	}
instance Show NonLinearFunction where
	show (NonLinearFunction name _) = "NonLinearFunction: " ++ name

find :: Foldable t => (a -> Bool) -> t a -> Maybe a
find condition =
	let fn candidate result
		| condition candidate = Just candidate
		| otherwise = result
	in foldr fn Nothing

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
squaredError (expected, actual) = (** 2) $ actual - expected

meanSquaredError :: CostFunction
meanSquaredError =
	let fn = mean . (map squaredError) . (uncurry zip)
	in CostFunction "MSE" fn
