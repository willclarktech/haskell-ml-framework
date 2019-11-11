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
	let
		r = round n
		prefix = replicate r 0
		suffix = replicate (9 - r) 0
	in prefix ++ (1:suffix)
processMnistOutput _ = error "Unrecognised output format"

processMnistTrainDataFile :: String -> ([[Float]], [[Float]])
processMnistTrainDataFile file =
	let
		trainData = processMnistTrainData $ processCsv file
		(rawOutputs, inputs) = unzip trainData
		outputs = map (processMnistOutput . fst) trainData
		-- outputs = map (map (\o -> if o == 5.0 then 1.0 else 0.0)) rawOutputs
	in (inputs, outputs)
