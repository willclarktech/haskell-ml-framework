module Utils where

split :: Char -> String -> [String]
split _ "" = [""]
split delimiter (c:cs)
	| c == delimiter = "" : rest
	| otherwise = (c : head rest) : tail rest
	where rest = split delimiter cs

-- Drops carriage return at end of line
processLine :: String -> [Float]
processLine = (map read) . (split ',') . init

-- Drops headers
processCsv :: String -> [[Float]]
processCsv = (map processLine) . tail . lines

-- MNIST train data has the correct label (0–9) followed by 784 pixel values
processMnistTrainData :: [[Float]] -> [([Float], [Float])]
processMnistTrainData = map (splitAt 1)

-- Converts MNIST numerical value to one-hot array
processMnistOutput :: [Float] -> [Float]
processMnistOutput (n:[]) =
	let r = round n
	in (replicate r 0) ++ [1] ++ (replicate (9 - r) 0)
processMnistOutput _ = error "Unrecognised output format"

processMnistTrainDataFile :: String -> ([[Float]], [[Float]])
processMnistTrainDataFile file =
	let
		trainData = processMnistTrainData $ processCsv file
		inputs = map snd trainData
		outputs = map fst trainData
	in (inputs, outputs)
