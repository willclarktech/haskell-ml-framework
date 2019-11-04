module Test where

import Layer

approxEqual :: Float -> Float -> Bool
approxEqual m n = abs (m - n) < 0.000001

allApproxEqual :: [Float] -> [Float] -> Bool
allApproxEqual ms ns
	| length ms == length ns = all (== True) $ zipWith approxEqual ms ns
	| otherwise = False

allDeepApproxEqual :: [[Float]] -> [[Float]] -> Bool
allDeepApproxEqual ms ns
	| length ms == length ns = all (== True) $ zipWith allApproxEqual ms ns
	| otherwise = False

maybeAllDeepApproxEqual :: Maybe [[Float]] -> Maybe [[Float]] -> Bool
maybeAllDeepApproxEqual Nothing Nothing = True
maybeAllDeepApproxEqual (Just maybeDeepFloats1) (Just maybeDeepFloats2) = allDeepApproxEqual maybeDeepFloats1 maybeDeepFloats2
maybeAllDeepApproxEqual expected result = False

layerApproxEqual :: Layer -> Layer -> Bool
layerApproxEqual (LinearLayer activations1 inputs1 weights1 biases1) (LinearLayer activations2 inputs2 weights2 biases2) =
	maybeAllDeepApproxEqual activations1 activations2
	&& maybeAllDeepApproxEqual inputs1 inputs2
	&& allDeepApproxEqual weights1 weights2
	&& allApproxEqual biases1 biases2
layerApproxEqual (NonLinearLayer activations1 inputs1 fn1) (NonLinearLayer activations2 inputs2 fn2) =
	maybeAllDeepApproxEqual activations1 activations2
	&& maybeAllDeepApproxEqual inputs1 inputs2
	&& fn1 == fn2
layerApproxEqual _ _ = error "Not implemented"

layersApproxEqual :: [Layer] -> [Layer] -> Bool
layersApproxEqual layer1 layer2
	| length layer1 == length layer2 = all (== True) $ zipWith layerApproxEqual layer1 layer2
	| otherwise = False

createError :: (Show a) => String -> a -> a -> b
createError name expected result = error $ "Test " ++ name ++ " failed: expected " ++ show expected ++ " got " ++ show result

check :: (Show a) => (a -> a -> Bool) -> String -> a -> a -> String
check condition name expected result = if condition expected result
	then "Passed: " ++ name
	else createError name expected result

checkApproxEqual :: String -> Float -> Float -> String
checkApproxEqual = check approxEqual

checkAllApproxEqual :: String -> [Float] -> [Float] -> String
checkAllApproxEqual = check allApproxEqual

checkAllDeepApproxEqual :: String -> [[Float]] -> [[Float]] -> String
checkAllDeepApproxEqual = check allDeepApproxEqual

checkLayerApproxEqual :: String -> Layer -> Layer -> String
checkLayerApproxEqual = check layerApproxEqual

checkLayersApproxEqual :: String -> [Layer] -> [Layer] -> String
checkLayersApproxEqual = check layersApproxEqual
