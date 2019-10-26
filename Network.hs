module Network where

import System.Random
import Layer

type Network = [Layer]

getOutputWidth :: Width -> Network -> Width
getOutputWidth inputWidth [] = inputWidth
getOutputWidth inputWidth network =
	case last network of
		LinearLayer weights _ -> length weights
		NonLinearLayer _ -> getOutputWidth inputWidth $ init network

appendLayer :: Width -> (StdGen, Network) -> LayerSpecification -> (StdGen, Network)
appendLayer inputWidth (g, network) specification =
	let
		previousWidth = getOutputWidth inputWidth network
		newLayer = createLayer g previousWidth specification
		newG = snd $ next g
	in (newG, network ++ [newLayer])

createNetwork :: StdGen -> Width -> [LayerSpecification] -> Network
createNetwork g inputWidth layerSpecifications =
	let (_, network) = foldl (appendLayer inputWidth) (g, []) layerSpecifications
	in network

forwardPropagateInput :: Network -> [Activation] -> [Activation]
forwardPropagateInput network input = foldl applyLayer input network
